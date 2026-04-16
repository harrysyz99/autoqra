"""
Atomic-edit neighborhoods on the discrete configuration space (Eq. 11).

An *atomic edit* changes exactly one variable q_ℓ or r_ℓ to an adjacent
value on its ladder. The atomic distance ``d_atom(C, C')`` counts the
minimum number of such edits transforming C into C'.

These neighborhoods power both the Phase II local-search offspring
generation and the Phase III trust-region construction.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from autoqra.core.config import ConfigEncoding


def atomic_distance(
    q1: Sequence[int],
    r1: Sequence[int],
    q2: Sequence[int],
    r2: Sequence[int],
    enc: ConfigEncoding,
) -> int:
    """Atomic edit distance ``d_atom(C, C')`` between two configurations."""
    dist = 0
    for i in range(len(q1)):
        if q1[i] != q2[i]:
            dist += abs(enc.Q.index(q1[i]) - enc.Q.index(q2[i]))
        if r1[i] != r2[i]:
            dist += abs(enc.R.index(r1[i]) - enc.R.index(r2[i]))
    return dist


def generate_k_nearest_atomic_neighbors(
    q: List[int],
    r: List[int],
    enc: ConfigEncoding,
    k: int = 5,
) -> List[Tuple[List[int], List[int]]]:
    """Generate up to *k* 1-edit-distance neighbors of (q, r).

    Enumerates all feasible single-step ladder moves, deduplicates, and
    samples *k* of them uniformly at random (all moves returned when
    fewer than *k* unique neighbors exist).
    """
    neighbors: List[Tuple[List[int], List[int]]] = []

    for l in range(len(q)):
        # q ladder moves
        idx_q = enc.Q.index(q[l])
        for new_idx in (idx_q - 1, idx_q + 1):
            if 0 <= new_idx < len(enc.Q):
                q_new = list(q)
                q_new[l] = enc.Q[new_idx]
                neighbors.append((q_new, list(r)))

        # r ladder moves
        idx_r = enc.R.index(r[l])
        for new_idx in (idx_r - 1, idx_r + 1):
            if 0 <= new_idx < len(enc.R):
                r_new = list(r)
                r_new[l] = enc.R[new_idx]
                neighbors.append((list(q), r_new))

    unique = list(set((tuple(qn), tuple(rn)) for qn, rn in neighbors))
    unique_list = [(list(qn), list(rn)) for qn, rn in unique]

    if len(unique_list) <= k:
        return unique_list
    return random.sample(unique_list, k)


def generate_atomic_neighbors(
    q: List[int],
    r: List[int],
    enc: ConfigEncoding,
    delta: int,
) -> List[Tuple[List[int], List[int]]]:
    """Enumerate or sample configurations within atomic-edit distance δ.

    For small δ (≤ 2) this enumerates the neighborhood exactly. For larger
    δ, it draws a random sample of at most min(200, 2^δ) configurations
    to keep the cost bounded.
    """
    if delta == 0:
        return [(list(q), list(r))]

    if delta <= 2:
        neighbors = {(tuple(q), tuple(r))}

        for l in range(len(q)):
            idx_q = enc.Q.index(q[l])
            for new_idx in (idx_q - 1, idx_q + 1):
                if 0 <= new_idx < len(enc.Q):
                    q_new = list(q)
                    q_new[l] = enc.Q[new_idx]
                    neighbors.add((tuple(q_new), tuple(r)))

            idx_r = enc.R.index(r[l])
            for new_idx in (idx_r - 1, idx_r + 1):
                if 0 <= new_idx < len(enc.R):
                    r_new = list(r)
                    r_new[l] = enc.R[new_idx]
                    neighbors.add((tuple(q), tuple(r_new)))

        if delta == 2:
            level1 = list(neighbors)
            for q1_t, r1_t in level1:
                q1 = list(q1_t)
                r1 = list(r1_t)
                for l in range(len(q1)):
                    idx_q = enc.Q.index(q1[l])
                    for new_idx in (idx_q - 1, idx_q + 1):
                        if 0 <= new_idx < len(enc.Q):
                            q_new = list(q1)
                            q_new[l] = enc.Q[new_idx]
                            neighbors.add((tuple(q_new), tuple(r1)))
                    idx_r = enc.R.index(r1[l])
                    for new_idx in (idx_r - 1, idx_r + 1):
                        if 0 <= new_idx < len(enc.R):
                            r_new = list(r1)
                            r_new[l] = enc.R[new_idx]
                            neighbors.add((tuple(q1), tuple(r_new)))

        return [(list(qt), list(rt)) for qt, rt in neighbors]

    # Large δ: random sampling.
    samples: List[Tuple[List[int], List[int]]] = [(list(q), list(r))]
    for _ in range(min(200, 2 ** delta)):
        q_s = list(q)
        r_s = list(r)
        budget = delta
        while budget > 0 and random.random() < 0.7:
            l = random.randrange(len(q_s))
            if random.random() < 0.5:
                idx = enc.Q.index(q_s[l])
                new_idx = idx + random.choice((-1, 1))
                if 0 <= new_idx < len(enc.Q):
                    q_s[l] = enc.Q[new_idx]
                    budget -= 1
            else:
                idx = enc.R.index(r_s[l])
                new_idx = idx + random.choice((-1, 1))
                if 0 <= new_idx < len(enc.R):
                    r_s[l] = enc.R[new_idx]
                    budget -= 1
        samples.append((q_s, r_s))
    return samples
