"""
Multi-objective optimization utilities: Pareto sorting, crowding distance,
and hypervolume for 2D minimization problems.

These primitives support both the NSGA-II selection step in Phase II and
the Pareto front extraction used to warm-start Phase III.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def non_dominated_sort_constrained(
    points: List[Tuple[float, float]],
    feasible: List[bool],
) -> List[List[int]]:
    """Constrained NSGA-II fast non-dominated sorting for 2D minimization.

    Domination rules (Deb et al., 2002):
      1. A feasible solution always dominates an infeasible one.
      2. Among feasible solutions: standard Pareto dominance applies.
      3. Among infeasible solutions: lower constraint violation preferred.

    Parameters
    ----------
    points : list[tuple[float, float]]
        Two-dimensional objective vectors (both objectives minimized).
    feasible : list[bool]
        Feasibility indicator for each candidate.

    Returns
    -------
    fronts : list[list[int]]
        List of Pareto fronts, each front being a list of indices.
    """
    n = len(points)
    S: List[List[int]] = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts: List[List[int]] = [[]]

    def dominates(p: int, q: int) -> bool:
        p_feasible, q_feasible = feasible[p], feasible[q]
        if p_feasible and not q_feasible:
            return True
        if not p_feasible and q_feasible:
            return False
        better_or_eq = (
            points[p][0] <= points[q][0] and points[p][1] <= points[q][1]
        )
        strictly_better = (
            points[p][0] < points[q][0] or points[p][1] < points[q][1]
        )
        return better_or_eq and strictly_better

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(p, q):
                S[p].append(q)
            elif dominates(q, p):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def non_dominated_sort(
    points: List[Tuple[float, float]],
) -> List[List[int]]:
    """Unconstrained NSGA-II sorting (all candidates treated as feasible)."""
    return non_dominated_sort_constrained(points, [True] * len(points))


def crowding_distance(
    points: List[Tuple[float, float]], indices: List[int]
) -> Dict[int, float]:
    """Crowding distance assignment for 2D minimization.

    Boundary points receive infinite distance to preserve diversity at the
    extremes of each objective.
    """
    if not indices:
        return {}

    d = {i: 0.0 for i in indices}
    for m in range(2):
        sorted_idx = sorted(indices, key=lambda i: points[i][m])
        d[sorted_idx[0]] = float("inf")
        d[sorted_idx[-1]] = float("inf")

        vmin = points[sorted_idx[0]][m]
        vmax = points[sorted_idx[-1]][m]
        denom = (vmax - vmin) if vmax != vmin else 1.0

        for j in range(1, len(sorted_idx) - 1):
            prev_v = points[sorted_idx[j - 1]][m]
            next_v = points[sorted_idx[j + 1]][m]
            d[sorted_idx[j]] += (next_v - prev_v) / denom
    return d


def hypervolume_2d(
    pareto_points: List[Tuple[float, float]],
    ref: Tuple[float, float],
) -> float:
    """Hypervolume indicator for 2D minimization against a reference point.

    Requires the input to be a non-dominated set in the 2D objective space.
    The reference point should dominate (in the minimization sense) the
    worst point of the front.
    """
    if not pareto_points:
        return 0.0

    pts = sorted(pareto_points, key=lambda x: x[0])
    hv = 0.0
    prev_x = ref[0]
    prev_y = ref[1]
    for x, y in reversed(pts):
        width = max(0.0, prev_x - x)
        height = max(0.0, prev_y - y)
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x
        prev_y = min(prev_y, y)
    return hv
