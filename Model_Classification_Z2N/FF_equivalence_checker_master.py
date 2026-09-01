"""
FF_equivalence_checker_master.py

Classification of order-two T-folds on the SO(12) lattice at the free fermionic point.

Consider models with basis vectors of 40 boundary condition components.  
The build_basis_* select the point group.

Fermion conventions of the 40 boundary condition components:

  index   0, 1      psi^mu                 |  20,21      psibar^mu
          2 - 7     chi^1..chi^6           |  22 - 27    chibar^1..chibar^6
          8 -13     y^1..y^6               |  28 - 33    ybar^1..ybar^6
         14 -19     w^1..w^6               |  34 - 39    wbar^1..wbar^6

Point groups supported
----------------------
This module used in every classify_*.py script the creates the 
Z2^N model classification table.  The most complicated cases of Z2L_2_Z2R_2 and Z2L_Z2R_Z2 import it directly

The build_basis_* helpers below also cover the remaining order-two point groups (Z2, Z2_2, Z2L, Z2L_2,
Z2L_Z2R) for reuse elsewhere -- those five are solved analytically.

Parameter conventions:
----------------------------------
  Z2L_Z2R_Z2   basis {1, S, Sbar, B1, B1b, B2b2}
               parameters n12, N ; k12, K ; m3456      (k = nbar, K = Nbar)
  Z2L_2_Z2R_2  basis {1, S, Sbar, B1, B2, B1b, B2b}
               parameters n12, N, m3456, M ; k12, K, l3456, L
                                                       (k = nbar, l = mbar,
                                                        K = Nbar, L = Mbar)
  Z2L_Z2       basis {1, S, Sbar, B1, B2b2}            parameters n12, N, m3456
  Z2L_2_Z2R    basis {1, S, Sbar, B1, B2, B1b}          parameters n12, N, m3456, M, k12, K

Equivalence relations (section 4.1)
-----------------------------------
  E1  GL(|B|;Z) changes of basis   -> handled by comparing additive sets Xi
  E2  y^i <-> w^i, per direction, left and right independently
  E3  permutation of the holomorphic or anti-holomorphic indices

E2 and E3 generate G = G_L x G_R with an S6 permutation of the six directions and then Z2^6 for the y<->w swaps, 
this gives |G_L| = |G_R| = 6! * 2^6 = 46080.
find_equivalence() essentially searches the 46080^2 possibilities
with g_L initially checked and g_R checked once a g_L is found so more efficient.

Modular invariance of the basis vectors, beta, is imposed:

    beta_a . beta_a = 0 mod 4   <=>  tr_L - tr_R = 0 mod 8
    beta_a . beta_b = 0 mod 2   <=>  ov_L - ov_R = 0 mod 4
    beta_a n beta_b n beta_c n beta_d = 0 mod 1
                                
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

POW40 = (1 << np.arange(40, dtype=np.int64))
POW20 = (1 << np.arange(20, dtype=np.int64))

BASIS_NAMES = {
    "Z2":          ["1", "S", "Sb", "B11b"],
    "Z2_2":        ["1", "S", "Sb", "B11b", "B22b"],
    "Z2L":         ["1", "S", "Sb", "B1"],
    "Z2L_2":       ["1", "S", "Sb", "B1", "B2"],
    "Z2L_Z2R":     ["1", "S", "Sb", "B1", "B1b"],
    "Z2L_Z2":      ["1", "S", "Sb", "B1", "B2b2"],
    "Z2L_2_Z2R":   ["1", "S", "Sb", "B1", "B2", "B1b"],
    "Z2L_Z2R_Z2":  ["1", "S", "Sb", "B1", "B1b", "B2b2"],
    "Z2L_2_Z2R_2": ["1", "S", "Sb", "B1", "B2", "B1b", "B2b"],
}


# --------------------------------------------------------------------------
# basis vectors
# --------------------------------------------------------------------------
def make_one() -> np.ndarray:
    return np.ones(40, dtype=np.uint8)


def make_S() -> np.ndarray:
    v = np.zeros(40, dtype=np.uint8); v[:8] = 1; return v


def make_Sbar() -> np.ndarray:
    v = np.zeros(40, dtype=np.uint8); v[20:28] = 1; return v


def make_B1(n1: int, n2: int, N: Sequence[int]) -> np.ndarray:
    """B1 = [b1 + n.e + N.ebar],  n = (n12, 0^4)."""
    v = np.zeros(40, dtype=np.uint8)
    v[4:8] = 1
    v[8:10] = [n1, n2]
    v[10:14] = 1
    v[14:16] = [n1, n2]
    v[28:34] = N
    v[34:40] = N
    return v


def make_B2(m: Sequence[int], M: Sequence[int]) -> np.ndarray:
    """B2 = [b2 + m.e + M.ebar],  m = (0^2, m3456)."""
    m3, m4, m5, m6 = m
    v = np.zeros(40, dtype=np.uint8)
    v[2:4] = 1; v[6:8] = 1
    v[8:10] = [1, 1]
    v[10:12] = [m3, m4]
    v[12:14] = [1 - m5, 1 - m6]
    v[16:18] = [m3, m4]
    v[18:20] = [m5, m6]
    v[28:34] = M
    v[34:40] = M
    return v


def make_Bb1(k1: int, k2: int, K: Sequence[int]) -> np.ndarray:
    """B1bar, with k12 = nbar12 and K = Nbar."""
    v = np.zeros(40, dtype=np.uint8)
    v[8:14] = K
    v[14:20] = K
    v[24:28] = 1
    v[28:30] = [k1, k2]
    v[30:34] = 1
    v[34:36] = [k1, k2]
    return v


def make_Bb2(l: Sequence[int], L: Sequence[int]) -> np.ndarray:
    """B2bar, with l3456 = mbar3456 and L = Mbar."""
    l3, l4, l5, l6 = l
    v = np.zeros(40, dtype=np.uint8)
    v[8:14] = L
    v[14:20] = L
    v[22:24] = 1; v[26:28] = 1
    v[28:30] = [1, 1]
    v[30:32] = [l3, l4]
    v[32:34] = [1 - l5, 1 - l6]
    v[36:38] = [l3, l4]
    v[38:40] = [l5, l6]
    return v


def make_B2b2(m: Sequence[int], mb: Optional[Sequence[int]] = None) -> np.ndarray:
    """Symmetric twist B_{2 2bar}."""
    if mb is None:
        mb = m
    m3, m4, m5, m6 = m
    b3, b4, b5, b6 = mb
    v = np.zeros(40, dtype=np.uint8)
    v[2:4] = 1; v[6:8] = 1
    v[8:14] = [1, 1, m3, m4, 1 - m5, 1 - m6]
    v[14:20] = [0, 0, m3, m4, m5, m6]
    v[22:24] = 1; v[26:28] = 1
    v[28:34] = [1, 1, b3, b4, 1 - b5, 1 - b6]
    v[34:40] = [0, 0, b3, b4, b5, b6]
    return v


def make_B1b1(n1: int, n2: int) -> np.ndarray:
    """Symmetric twist B_{1 1bar} = [b_{1 1bar} + n.e + n.ebar], n = (n12, 0^4)."""
    v = np.zeros(40, dtype=np.uint8)
    for off in (0, 20):
        v[off + 4:off + 8] = 1                 # chi^3..chi^6
        v[off + 8] = n1; v[off + 9] = n2       # y^1, y^2
        v[off + 10:off + 14] = 1               # y^3..y^6
        v[off + 14] = n1; v[off + 15] = n2     # w^1, w^2
    return v


# --------------------------------------------------------------------------
# basis builders, one per point group
# --------------------------------------------------------------------------
def build_basis_Z2L_Z2R_Z2(n12, N, k12, K, m3456) -> np.ndarray:
    return np.vstack([
        make_one(), make_S(), make_Sbar(),
        make_B1(n12[0], n12[1], N),
        make_Bb1(k12[0], k12[1], K),
        make_B2b2(m3456),
    ]).astype(np.uint8) % 2


def build_basis_Z2L_2_Z2R_2(n12, N, m3456, M, k12, K, l3456, L) -> np.ndarray:
    return np.vstack([
        make_one(), make_S(), make_Sbar(),
        make_B1(n12[0], n12[1], N),
        make_B2(m3456, M),
        make_Bb1(k12[0], k12[1], K),
        make_Bb2(l3456, L),
    ]).astype(np.uint8) % 2


def build_basis_Z2_2(n12, m3456) -> np.ndarray:
    """{1, S, Sbar, B_{1 1bar}, B_{2 2bar}} for the symmetric Z2^2 point group."""
    return np.vstack([make_one(), make_S(), make_Sbar(),
                      make_B1b1(n12[0], n12[1]),
                      make_B2b2(m3456)]).astype(np.uint8) % 2


def build_basis_Z2L_2(n12, N, m3456, M) -> np.ndarray:
    """{1, S, Sbar, B1, B2} for Z2L^2."""
    return np.vstack([make_one(), make_S(), make_Sbar(),
                      make_B1(n12[0], n12[1], N),
                      make_B2(m3456, M)]).astype(np.uint8) % 2


def build_basis_Z2L_2_Z2R(n12, N, m3456, M, k12, K) -> np.ndarray:
    """{1, S, Sbar, B1, B2, B1bar} for Z2L^2 x Z2R."""
    return np.vstack([make_one(), make_S(), make_Sbar(),
                      make_B1(n12[0], n12[1], N),
                      make_B2(m3456, M),
                      make_Bb1(k12[0], k12[1], K)]).astype(np.uint8) % 2


def build_basis_Z2L_Z2(n12, N, m3456) -> np.ndarray:
    """{1, S, Sbar, B1, B_{2 2bar}} for Z2L x Z2."""
    return np.vstack([make_one(), make_S(), make_Sbar(),
                      make_B1(n12[0], n12[1], N),
                      make_B2b2(m3456)]).astype(np.uint8) % 2


def build_basis_Z2L(n12, N) -> np.ndarray:
    """{1, S, Sbar, B1} for Z2L."""
    return np.vstack([make_one(), make_S(), make_Sbar(),
                      make_B1(n12[0], n12[1], N)]).astype(np.uint8) % 2


def build_basis_Z2(n12) -> np.ndarray:
    """{1, S, Sbar, B_{1 1bar}} for the symmetric Z2."""
    return np.vstack([make_one(), make_S(), make_Sbar(),
                      make_B1b1(n12[0], n12[1])]).astype(np.uint8) % 2


def build_basis_Z2L_Z2R(n12, N, k12, K) -> np.ndarray:
    """{1, S, Sbar, B1, B1bar} for Z2L x Z2R."""
    return np.vstack([make_one(), make_S(), make_Sbar(),
                      make_B1(n12[0], n12[1], N),
                      make_Bb1(k12[0], k12[1], K)]).astype(np.uint8) % 2


def additive_set(basis: np.ndarray) -> np.ndarray:
    out = np.zeros((1, 40), dtype=np.uint8)
    for row in basis:
        out = np.concatenate([out, (out + row) % 2], axis=0)
    return out


def encode(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.int64) @ POW40


# --------------------------------------------------------------------------
# modular invariance
# --------------------------------------------------------------------------
def modular_invariant(basis: np.ndarray) -> bool:
    n = basis.shape[0]
    b = basis.astype(np.int64)
    for a in range(n):
        if (b[a, :20].sum() - b[a, 20:].sum()) % 8:
            return False
    for a in range(n):
        for c in range(a + 1, n):
            if (b[a, :20] @ b[c, :20] - b[a, 20:] @ b[c, 20:]) % 4:
                return False
    for quad in itertools.combinations(range(n), 4):
        if int(np.prod(b[list(quad)], axis=0).sum()) % 2:
            return False
    return True


# --------------------------------------------------------------------------
# the g_L and g_R for E2 and E3 logic
# --------------------------------------------------------------------------
def _half_targets(pi, fl) -> np.ndarray:
    t = np.arange(20, dtype=np.int64)
    for i in range(6):
        t[2 + i] = 2 + pi[i]
        t[8 + i] = (14 if fl[i] else 8) + pi[i]
        t[14 + i] = (8 if fl[i] else 14) + pi[i]
    return t


def _build_half_group():
    params, W = [], []
    for pi in itertools.permutations(range(6)):
        for fl in itertools.product((0, 1), repeat=6):
            params.append((pi, fl))
            W.append(POW20[_half_targets(pi, fl)])
    return np.array(W, dtype=np.int64), params


WH, PARAMS = _build_half_group()                     # 46080 x 20
_WEIGHTS = np.random.default_rng(12345).integers(1, 2 ** 20, size=8192)


def apply_g(v: np.ndarray, gL, gR) -> np.ndarray:
    """Explicit action of g = (g_L, g_R) on a 40-vector."""
    out = np.zeros(40, dtype=np.uint8)
    out[[0, 1, 20, 21]] = v[[0, 1, 20, 21]]
    for off, (pi, fl) in ((0, gL), (20, gR)):
        for i in range(6):
            out[off + 2 + pi[i]] = v[off + 2 + i]
            hi, lo = (14, 8) if fl[i] else (8, 14)
            out[off + hi + pi[i]] = v[off + 8 + i]
            out[off + lo + pi[i]] = v[off + 14 + i]
    return out


def _right_invariant(xi: np.ndarray) -> np.ndarray:
    """
    Looking for g_R once a g_L is found
    """
    trR = xi[:, 20:].astype(np.int64).sum(1)
    chiR = xi[:, 22:28].astype(np.int64).sum(1)
    return trR * 8 + chiR


def prepare_starting_model(xi: np.ndarray, basis: Optional[np.ndarray] = None) -> Dict:
    """
    Get the initial model's additive set ready to be transformed and matched
    against a target. Passing the basis caches its code tables too, which is
    what makes repeated find_equivalence calls against the same model cheap.
    """
    left = xi[:, :20].astype(np.int64) @ WH.T          # left half of Xi under every g_L
    fingerprint_per_gL = np.sort(left * 256 + _right_invariant(xi)[:, None], axis=0)
    weights = _WEIGHTS[:xi.shape[0]]
    model = dict(xi=xi, fingerprint_per_gL=fingerprint_per_gL,
                hash_per_gL=(fingerprint_per_gL * weights[:, None]).sum(0))
    if basis is not None:
        model["basis"] = basis
        model["basis_left"] = basis[:, :20].astype(np.int64)
        model["basis_right_per_gR"] = basis[:, 20:].astype(np.int64) @ WH.T
    return model


def prepare_target_model(xi: np.ndarray) -> Dict:
    """Get the target model's additive set ready to be matched against."""
    left = xi[:, :20].astype(np.int64) @ POW20
    right = xi[:, 20:].astype(np.int64) @ POW20
    right_halves_seen: Dict[int, List[int]] = {}
    for l, r in zip(left.tolist(), right.tolist()):
        right_halves_seen.setdefault(l, []).append(r)
    right_halves_allowed = {l: np.array(sorted(set(rs)), dtype=np.int64)
                            for l, rs in right_halves_seen.items()}
    fingerprint = np.sort(left * 256 + _right_invariant(xi))
    weights = _WEIGHTS[:xi.shape[0]]
    return dict(xi=xi, fingerprint=fingerprint, right_halves_allowed=right_halves_allowed,
                hash=int((fingerprint * weights).sum()), codes=np.sort(encode(xi)))


def find_equivalence(source_model: Dict, target_model: Dict, basis: Optional[np.ndarray] = None,
                     max_gL: int = 200000):
    """
    Search all of G_L x G_R for a relabelling g with g(Xi_source) = Xi_target.
    source_model from prepare_starting_model, target_model from
    prepare_target_model.  Returns g = (g_L, g_R) as ((piL,flL),(piR,flR)), or None.
    """
    gL_candidates = np.flatnonzero(source_model["hash_per_gL"] == target_model["hash"])
    gL_candidates = [gL for gL in gL_candidates if np.array_equal(
        source_model["fingerprint_per_gL"][:, gL], target_model["fingerprint"])]
    if not gL_candidates:
        return None
    basis_left = source_model.get("basis_left")
    basis_right_per_gR = source_model.get("basis_right_per_gR")
    if basis_left is None or basis_right_per_gR is None:   # not cached: compute now
        if basis is None:
            raise ValueError("pass the basis to prepare_starting_model or to find_equivalence")
        basis_left = basis[:, :20].astype(np.int64)
        basis_right_per_gR = basis[:, 20:].astype(np.int64) @ WH.T
    for gL in gL_candidates[:max_gL]:
        allowed = [target_model["right_halves_allowed"].get(int(code)) for code in basis_left @ WH[gL]]
        if any(a is None for a in allowed):
            continue                                    # this g_L can't map every basis vector into the target
        gR_ok = None
        for i in sorted(range(len(allowed)), key=lambda j: len(allowed[j])):
            pos = np.searchsorted(allowed[i], basis_right_per_gR[i])
            np.clip(pos, 0, len(allowed[i]) - 1, out=pos)
            match = allowed[i][pos] == basis_right_per_gR[i]
            gR_ok = match if gR_ok is None else (gR_ok & match)
            if not gR_ok.any():
                gR_ok = None
                break
        if gR_ok is None:
            continue
        g_L, g_R = PARAMS[gL], PARAMS[int(np.flatnonzero(gR_ok)[0])]
        positions = np.concatenate([_half_targets(*g_L), 20 + _half_targets(*g_R)])
        transformed = np.zeros_like(source_model["xi"])
        transformed[:, positions] = source_model["xi"]     # apply g
        if np.array_equal(np.sort(encode(transformed)), target_model["codes"]):
            return g_L, g_R
    return None


def trace_signature(xi: np.ndarray) -> Tuple:
    """(tr_L, tr_R) per vector in Xi, sorted -- a relabelling invariant used to bucket models before the full search."""
    tr_L = xi[:, :20].sum(1)
    tr_R = xi[:, 20:].sum(1)
    return tuple(sorted(zip(tr_L.tolist(), tr_R.tolist())))


# --------------------------------------------------------------------------
# E1: express a twist vector in a given basis
# --------------------------------------------------------------------------
def basis_coefficients(basis: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    """Coefficients c with c.basis = target (mod 2), or None if target isn't in the span of basis."""
    n = basis.shape[0]
    augmented = np.concatenate([basis.copy() % 2, np.eye(n, dtype=np.uint8)], axis=1)
    pivot_cols, rank = [], 0
    for col in range(40):
        pivot_row = next((i for i in range(rank, n) if augmented[i, col]), None)
        if pivot_row is None:
            continue
        augmented[[rank, pivot_row]] = augmented[[pivot_row, rank]]
        for i in range(n):
            if i != rank and augmented[i, col]:
                augmented[i] = (augmented[i] + augmented[rank]) % 2
        pivot_cols.append(col); rank += 1
        if rank == n:
            break
    remainder = target.copy() % 2
    coefficients = np.zeros(n, dtype=np.uint8)
    for i, col in enumerate(pivot_cols):
        if remainder[col]:
            remainder = (remainder + augmented[i, :40]) % 2
            coefficients = (coefficients + augmented[i, 40:]) % 2
    return None if remainder.any() else coefficients


def e1_change_of_basis_note(src_twists, src_names, dst_basis, dst_names, g) -> str:
    """How the relabelled source twists sit inside the target basis ('1 + S + Sb' shown as E)."""
    notes = []
    for name, vec in zip(src_names, src_twists):
        coefficients = basis_coefficients(dst_basis, apply_g(vec, g[0], g[1]))
        if coefficients is None:
            return "?"
        terms = [dst_names[j] for j in range(len(coefficients)) if coefficients[j]]
        if all(x in terms for x in ("1", "S", "Sb")):
            terms = ["E"] + [x for x in terms if x not in ("1", "S", "Sb")]
        notes.append(f"{name} -> " + (" + ".join(terms) if terms else "0"))
    return "; ".join(notes)


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------
_PURE = {frozenset(): None, frozenset({3, 4, 5, 6}): "1",
         frozenset({1, 2, 5, 6}): "2", frozenset({1, 2, 3, 4}): "3"}
_PURE_Y = {"1": {3, 4, 5, 6}, "2": {1, 2, 5, 6}, "3": {1, 2, 3, 4}}


def twist_label(vec: np.ndarray) -> str:
    """'b1 + e12`3`4' -- a backtick marks an anti-holomorphic index."""
    left_twist = _PURE.get(frozenset(i + 1 for i in range(6) if vec[2 + i]), "?")
    right_twist = _PURE.get(frozenset(i + 1 for i in range(6) if vec[22 + i]), "?")
    if "?" in (left_twist, right_twist):
        return "<non standard>"
    pure_twist_vector = np.zeros(40, dtype=np.uint8)
    for twist, offset in ((left_twist, 0), (right_twist, 20)):
        if twist:
            for i in _PURE_Y[twist]:
                pure_twist_vector[offset + 2 + i - 1] = 1
                pure_twist_vector[offset + 8 + i - 1] = 1
    shift = (vec + pure_twist_vector) % 2
    shift_dirs = [[], []]
    for chirality, offset in enumerate((0, 20)):
        for i in range(6):
            if shift[offset + 8 + i] != shift[offset + 14 + i]:
                return "<not pure twist + shifts>"
            if shift[offset + 8 + i]:
                shift_dirs[chirality].append(i + 1)
    twist_name = (f"b{left_twist}" if left_twist else "") + (
        (f"`{right_twist}" if left_twist else f"b`{right_twist}") if right_twist else "")
    shift_label = "".join(map(str, shift_dirs[0])) + "".join("`" + str(i) for i in shift_dirs[1])
    return (twist_name or "E") + (" + e" + shift_label if shift_label else "")


def relabelling_note(g_half) -> str:
    """Human-readable form of one chirality's relabelling: a permutation plus which directions swap y<->w."""
    perm, flips = g_half
    moves = ", ".join(f"{i+1}->{perm[i]+1}" for i in range(6) if perm[i] != i)
    swapped = "".join(str(i + 1) for i in range(6) if flips[i])
    return (moves or "identity") + (f" ; y<->w on {swapped}" if swapped else "")


def describe_equivalence(src, dst, g, names) -> str:
    """Human-readable note written into the output csv, explaining why src is equivalent to dst."""
    return (f"duplicate of {dst['label']}. "
            f"E3/E2 permutations -- holomorphic: {relabelling_note(g[0])}; "
            f"anti-holomorphic: {relabelling_note(g[1])}. "
            f"E1 change of basis (E = 1+S+Sb): "
            f"{e1_change_of_basis_note(src['twists'], names[3:], dst['basis'], names, g)}")
