"""
High-fidelity real-task evaluator.

Dispatches a subprocess pipeline:

    train_sft.py  →  post_quant.py  →  eval_tasks.py

for each candidate configuration, caching the results by a SHA-1 hash of
(q, r) so that repeated evaluations are free. Supports two fidelity levels
(LF / HF) with distinct (sample_ratio, epochs) budgets.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


class RealTaskEvaluator:
    """Run SFT + quantization + lm-eval to obtain true downstream metrics.

    Parameters
    ----------
    preset : str
        Model preset name (e.g., 'qwen3-4b', 'llama3.1-8b').
    dataset : str
        SFT dataset identifier (e.g., 'alpaca', 'hc3').
    lf_sample_ratio, lf_epochs : float
        Low-fidelity SFT hyperparameters (fast, noisy).
    hf_sample_ratio, hf_epochs : float
        High-fidelity SFT hyperparameters (slow, precise).
    eval_task : str
        Downstream lm-eval task identifier (e.g., 'winogrande').
    eval_shots : int
        Few-shot count for lm-eval harness.
    output_root : Path | None
        Root directory where per-candidate working dirs are cached.
    load_in_4bit : bool
        Whether to pass --load_in_4bit to the SFT trainer.
    """

    def __init__(
        self,
        preset: str,
        dataset: str,
        lf_sample_ratio: float = 0.05,
        lf_epochs: float = 0.1,
        hf_sample_ratio: float = 0.2,
        hf_epochs: float = 0.5,
        eval_task: str = "",
        eval_shots: int = 0,
        output_root: Optional[Path] = None,
        load_in_4bit: bool = True,
    ) -> None:
        if not preset:
            raise ValueError(
                "preset must be provided when using real-task evaluation."
            )
        self.preset = preset
        self.dataset = dataset
        self.lf_sample_ratio = lf_sample_ratio
        self.lf_epochs = lf_epochs
        self.hf_sample_ratio = hf_sample_ratio
        self.hf_epochs = hf_epochs
        self.eval_task = eval_task
        self.eval_shots = eval_shots
        self.load_in_4bit = load_in_4bit

        self.base_dir = Path(output_root) if output_root else Path("results_real_eval")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.python = sys.executable or "python"
        self.cache: Dict[str, Dict[str, float]] = {}
        self.index_path = self.base_dir / "candidates_index.json"
        self.dir_index: Dict[str, str] = (
            json.loads(self.index_path.read_text())
            if self.index_path.exists()
            else {}
        )

    def _hash_config(self, q: Sequence[int], r: Sequence[int]) -> str:
        payload = json.dumps({"q": list(q), "r": list(r)}, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _save_index(self) -> None:
        try:
            with open(self.index_path, "w") as f:
                json.dump(self.dir_index, f, indent=2)
        except Exception:
            pass

    def _extract_accuracy(self, eval_json: Path) -> float:
        data = json.loads(eval_json.read_text())
        task_res = data.get("results", {}).get(self.eval_task, {})
        for key in ("acc", "acc_norm", "acc,none", "acc_norm,none",
                    "f1", "exact_match"):
            if key in task_res:
                return float(task_res[key])
        raise RuntimeError(
            f"No accuracy key found for task '{self.eval_task}' in {eval_json}"
        )

    def evaluate(
        self,
        q: Sequence[int],
        r: Sequence[int],
        high_fidelity: bool = False,
        generation: Optional[int] = None,
        cand_idx: Optional[int] = None,
        stage: str = "",
    ) -> Tuple[float, float]:
        """Evaluate a configuration, returning (performance, memory_bytes)."""
        sample_ratio = self.hf_sample_ratio if high_fidelity else self.lf_sample_ratio
        epochs = self.hf_epochs if high_fidelity else self.lf_epochs
        fidelity_suffix = "_HF" if high_fidelity else "_LF"

        base_key = self._hash_config(q, r)
        cache_key = base_key + fidelity_suffix

        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return cached["perf"], cached["mem"]

        dir_name = self.dir_index.get(
            cache_key, f"cand_{base_key[:12]}{fidelity_suffix.lower()}"
        )
        if cache_key not in self.dir_index:
            self.dir_index[cache_key] = dir_name
            self._save_index()

        work_dir = self.base_dir / dir_name
        metrics_path = work_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            self.cache[cache_key] = metrics
            return metrics["perf"], metrics["mem"]

        work_dir.mkdir(parents=True, exist_ok=True)
        config_path = work_dir / "qra_config.json"
        with open(config_path, "w") as f:
            json.dump({"q": list(q), "r": list(r)}, f, indent=2)

        # 1) SFT training.
        train_dir = work_dir / "sft"
        train_cmd = [
            self.python, "-m", "autoqra.training.sft",
            "--preset", self.preset,
            "--dataset", self.dataset,
            "--sample_ratio", str(sample_ratio),
            "--epochs", str(epochs),
            "--qra_config", str(config_path),
            "--output_dir", str(train_dir),
        ]
        if self.load_in_4bit:
            train_cmd.append("--load_in_4bit")
        subprocess.run(train_cmd, check=True)

        profile = json.loads((train_dir / "train_profile.json").read_text())
        mem_bytes = float(profile.get("peak_mem_bytes", 0.0))

        # 2) Merge adapter and save post-quant state.
        quant_dir = work_dir / "quant"
        subprocess.run(
            [
                self.python, "-m", "autoqra.training.post_quant",
                "--adapter_dir", str(train_dir / "final_model"),
                "--qra_config", str(config_path),
                "--preset", self.preset,
                "--out_dir", str(quant_dir),
            ],
            check=True,
        )

        # 3) lm-eval on the downstream task.
        eval_out = work_dir / "eval_results.json"
        subprocess.run(
            [
                self.python, "-m", "autoqra.experiments.eval_tasks",
                "--model_path", str(quant_dir),
                "--task", self.eval_task,
                "--shots", str(self.eval_shots),
                "--out", str(eval_out),
            ],
            check=True,
        )
        perf = self._extract_accuracy(eval_out)

        metrics = {"perf": perf, "mem": mem_bytes}
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self.cache[cache_key] = metrics
        return perf, mem_bytes
