"""Smoke tests for the public API of autoqra."""

import json
import tempfile
from pathlib import Path

import numpy as np

from autoqra import (
    AutoQRAConfig,
    ConfigEncoding,
    Importance,
    MemoryModel,
    ProxyEvaluator,
    AutoQRA,
)
from autoqra.core.pareto import (
    crowding_distance,
    hypervolume_2d,
    non_dominated_sort_constrained,
)
from autoqra.search.neighbors import (
    atomic_distance,
    generate_k_nearest_atomic_neighbors,
)
from autoqra.search.operators import repair_to_budget, warm_start_from_importance


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------
def test_config_encoding():
    enc = ConfigEncoding([2, 4, 8], [4, 8, 16])
    assert enc.s_q(2) == 0.0
    assert enc.s_q(8) == 1.0
    assert enc.round_Q(3.0) == 2
    assert enc.round_R(10.0) == 8


def test_importance_normalization():
    bb = np.array([1.0, 2.0, 3.0, 4.0])
    lr = np.array([4.0, 3.0, 2.0, 1.0])
    imp = Importance(bb, lr)
    assert imp.I_q.min() == 0.0 and imp.I_q.max() == 1.0
    assert imp.I_r.min() == 0.0 and imp.I_r.max() == 1.0


def test_memory_model_layer_breakdown():
    cfg = AutoQRAConfig(num_layers=4, Q=[2, 4, 8], R=[4, 8, 16])
    mem = MemoryModel(cfg)
    total = mem.total_memory_bytes([4] * 4, [8] * 4)
    assert total > 0


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------
def test_non_dominated_sort_constrained():
    pts = [(0.1, 0.5), (0.3, 0.2), (0.2, 0.4), (0.5, 0.1)]
    feasible = [True, True, True, True]
    fronts = non_dominated_sort_constrained(pts, feasible)
    assert sum(len(f) for f in fronts) == len(pts)


def test_crowding_distance_boundaries():
    pts = [(0.1, 0.4), (0.2, 0.3), (0.3, 0.2)]
    cd = crowding_distance(pts, [0, 1, 2])
    assert cd[0] == float("inf")
    assert cd[2] == float("inf")


def test_hypervolume_monotone():
    hv1 = hypervolume_2d([(0.5, 0.5)], (1.0, 1.0))
    hv2 = hypervolume_2d([(0.3, 0.3)], (1.0, 1.0))
    assert hv2 > hv1 > 0


# ---------------------------------------------------------------------------
# Search operators
# ---------------------------------------------------------------------------
def test_warm_start_shapes():
    enc = ConfigEncoding([2, 4, 8], [4, 8, 16])
    I_q = np.array([0.1, 0.5, 0.9, 0.3])
    I_r = np.array([0.2, 0.8, 0.4, 0.6])
    q, r = warm_start_from_importance(enc, I_q, I_r)
    assert len(q) == 4 and len(r) == 4
    assert all(qi in enc.Q for qi in q)
    assert all(ri in enc.R for ri in r)


def test_atomic_distance_zero_for_identical():
    enc = ConfigEncoding([2, 4, 8], [4, 8, 16])
    q = [4, 4, 8]
    r = [8, 16, 4]
    assert atomic_distance(q, r, q, r, enc) == 0


def test_neighbors_unique():
    enc = ConfigEncoding([2, 4, 8], [4, 8, 16])
    neighbors = generate_k_nearest_atomic_neighbors([4, 4], [8, 8], enc, k=10)
    serialized = {(tuple(q), tuple(r)) for q, r in neighbors}
    assert len(serialized) == len(neighbors)


def test_repair_brings_under_budget():
    cfg = AutoQRAConfig(num_layers=8, Q=[2, 4, 8], R=[4, 8, 16])
    enc = ConfigEncoding(cfg.Q, cfg.R)
    mem = MemoryModel(cfg)
    bb = np.linspace(0.1, 0.9, 8)
    lr = np.linspace(0.9, 0.1, 8)
    q = [8] * 8
    r = [16] * 8
    full_mem = mem.total_memory_bytes(q, r)
    budget = full_mem * 0.3
    q2, r2 = repair_to_budget(q, r, enc, bb, lr, mem, budget)
    assert mem.total_memory_bytes(q2, r2) <= budget


# ---------------------------------------------------------------------------
# Proxy evaluator and Phase II → III pipeline
# ---------------------------------------------------------------------------
def test_proxy_evaluator_returns_two_values():
    cfg = AutoQRAConfig(num_layers=4, Q=[2, 4, 8], R=[4, 8, 16])
    enc = ConfigEncoding(cfg.Q, cfg.R)
    mem = MemoryModel(cfg)
    imp = Importance(np.linspace(0.1, 0.9, 4), np.linspace(0.9, 0.1, 4))
    proxy = ProxyEvaluator(cfg, imp, mem, enc)
    score, mbytes = proxy.evaluate([4, 4, 4, 4], [8, 8, 8, 8], None)
    assert isinstance(score, float)
    assert mbytes > 0


def test_autoqra_end_to_end_proxy(tmp_path):
    nl = 8
    bb = np.random.RandomState(0).rand(nl)
    lr = np.random.RandomState(1).rand(nl)
    payload = {
        "backbone_metric_per_layer": {str(i): float(bb[i]) for i in range(nl)},
        "lora_metric_per_layer": {str(i): float(lr[i]) for i in range(nl)},
    }
    imp_json = tmp_path / "imp.json"
    imp_json.write_text(json.dumps(payload))

    cfg = AutoQRAConfig(
        num_layers=nl, Q=[2, 4, 8], R=[4, 8, 16],
        seed=42, budget_bytes=1e8,
    )
    out = tmp_path / "out"
    runner = AutoQRA(cfg, importance_json=str(imp_json))
    result = runner.run(
        outdir=out,
        phase2_kwargs=dict(
            pop_size=6, generations=2, promote_k=2, gamma=1.5,
            lf_eval_mode="proxy",
        ),
        phase3_alpha=0.6,
    )
    assert (out / "phase2_pareto.json").exists()
    assert isinstance(result["phase2"]["pareto"], list)
