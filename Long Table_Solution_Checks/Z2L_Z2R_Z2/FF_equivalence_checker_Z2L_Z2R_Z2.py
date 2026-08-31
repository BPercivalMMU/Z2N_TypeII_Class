"""
ff_equiv.py -- shared machinery for the Z2L x Z2R x Z2 classification.

Fermion layout of the 40-component boundary condition vectors
(identical to get_model_spectra_stats_all_classes.py):

  index   0, 1      psi^mu                 |  20,21      psibar^mu
          2 - 7     chi^1..chi^6           |  22 - 27    chibar^1..chibar^6
          8 -13     y^1..y^6               |  28 - 33    ybar^1..ybar^6
         14 -19     w^1..w^6               |  34 - 39    wbar^1..wbar^6

Equivalence relations (section 4.1)
-----------------------------------
  E1  GL(|B|;Z) changes of basis          -> handled by comparing the additive
                                             sets Xi, not the bases
  E2  y^i <-> w^i (per direction, either side independently)
  E3  permutation of the holomorphic, or of the anti-holomorphic, indices

E2 and E3 generate G = G_L x G_R with G_L = G_R = S_6 |x (Z_2)^6, of order
46080 each.  find_equivalence() searches all 46080^2 elements of G without
enumerating them, by screening g_L first on the multiset of left halves of Xi
and then g_R against the right halves that the choice of g_L allows.

Modular invariance (3.4) is imposed directly on the vectors, using the
Minkowskian product  alpha.beta = 1/2 alpha_L.beta_L - 1/2 alpha_R.beta_R :

    beta_a . beta_a = 0 mod 4        <=>  tr_L - tr_R = 0 mod 8
    beta_a . beta_b = 0 mod 2        <=>  ov_L - ov_R = 0 mod 4
    beta_a n beta_b n beta_c n beta_d = 0 mod 1
                                     <=>  #{positions where all four are 1} even
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

POW40 = (1 << np.arange(40, dtype=np.int64))
POW20 = (1 << np.arange(20, dtype=np.int64))


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
    v[4:8] = 1                       # chi^3..chi^6
    v[8:10] = [n1, n2]               # y^1, y^2
    v[10:14] = 1                     # y^3..y^6
    v[14:16] = [n1, n2]              # w^1, w^2
    v[28:34] = N                     # ybar
    v[34:40] = N                     # wbar
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


def make_B2b2(m: Sequence[int], mb: Optional[Sequence[int]] = None) -> np.ndarray:
    """Symmetric twist B_{2 2bar}, m = (m3,m4,m5,m6)."""
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


def build_basis(n12, N, k12, K, m3456) -> np.ndarray:
    """{1, S, Sbar, B1, B1bar, B_{2 2bar}} for the Z2L x Z2R x Z2 point group."""
    return np.vstack([
        make_one(), make_S(), make_Sbar(),
        make_B1(n12[0], n12[1], N),
        make_Bb1(k12[0], k12[1], K),
        make_B2b2(m3456),
    ]).astype(np.uint8) % 2


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
            ov = b[a, :20] @ b[c, :20] - b[a, 20:] @ b[c, 20:]
            if ov % 4:
                return False
    for quad in itertools.combinations(range(n), 4):
        if int(np.prod(b[list(quad)], axis=0).sum()) % 2:
            return False
    return True


# --------------------------------------------------------------------------
# the relabelling group acting on one chirality
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


WH, PARAMS = _build_half_group()          # 46080 x 20
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


def prep_source(xi: np.ndarray) -> Dict:
    """Data for the configuration we transform (the expensive side)."""
    L = xi[:, :20].astype(np.int64)
    allL = np.sort(L @ WH.T, axis=0)                    # (|Xi|, 46080)
    w = _WEIGHTS[:xi.shape[0]]
    return dict(xi=xi, allL=allL, hash_all=(allL * w[:, None]).sum(0))


def prep_target(xi: np.ndarray) -> Dict:
    """Data for the configuration we transform onto (cheap)."""
    lc = xi[:, :20].astype(np.int64) @ POW20
    rc = xi[:, 20:].astype(np.int64) @ POW20
    lut: Dict[int, np.ndarray] = {}
    tmp: Dict[int, List[int]] = {}
    for a, b in zip(lc.tolist(), rc.tolist()):
        tmp.setdefault(a, []).append(b)
    for a, v in tmp.items():
        lut[a] = np.array(sorted(set(v)), dtype=np.int64)
    w = _WEIGHTS[:xi.shape[0]]
    return dict(xi=xi, sorted_lc=np.sort(lc), lut=lut,
                target=int((np.sort(lc) * w).sum()),
                codes=np.sort(encode(xi)))


def find_equivalence(A: Dict, B: Dict, basis_a: np.ndarray,
                     max_gl: int = 20000):
    """
    Search all of G_L x G_R for g with g(Xi_A) = Xi_B.
    A must come from prep_source, B from prep_target.
    Returns ((piL, flL), (piR, flR)) or None.
    """
    cand = np.flatnonzero(A["hash_all"] == B["target"])
    cand = [c for c in cand if np.array_equal(A["allL"][:, c], B["sorted_lc"])]
    if not cand:
        return None
    bl = basis_a[:, :20].astype(np.int64)
    rc_all = basis_a[:, 20:].astype(np.int64) @ WH.T
    for gl in cand[:max_gl]:
        ok = np.ones(WH.shape[0], dtype=bool)
        for i in range(bl.shape[0]):
            allowed = B["lut"].get(int(bl[i] @ WH[gl]))
            if allowed is None:
                ok = None
                break
            ok &= np.isin(rc_all[i], allowed)
            if not ok.any():
                ok = None
                break
        if ok is None:
            continue
        gr = int(np.flatnonzero(ok)[0])
        gL, gR = PARAMS[gl], PARAMS[gr]
        img = np.stack([apply_g(v, gL, gR) for v in A["xi"]])
        if np.array_equal(np.sort(encode(img)), B["codes"]):   # verify
            return gL, gR
    return None


def fingerprint(xi: np.ndarray) -> Tuple:
    """Relabelling invariant used to bucket configurations before testing."""
    trL = xi[:, :20].sum(1)
    trR = xi[:, 20:].sum(1)
    return tuple(sorted(zip(trL.tolist(), trR.tolist())))


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------
_PURE = {frozenset(): None, frozenset({3, 4, 5, 6}): "1",
         frozenset({1, 2, 5, 6}): "2", frozenset({1, 2, 3, 4}): "3"}
_PURE_Y = {"1": {3, 4, 5, 6}, "2": {1, 2, 5, 6}, "3": {1, 2, 3, 4}}


def describe(vec: np.ndarray) -> str:
    """'b1 + e12`3`4' -- backtick marks an anti-holomorphic index."""
    aL = _PURE.get(frozenset(i + 1 for i in range(6) if vec[2 + i]), "?")
    aR = _PURE.get(frozenset(i + 1 for i in range(6) if vec[22 + i]), "?")
    if "?" in (aL, aR):
        return "<non standard>"
    pure = np.zeros(40, dtype=np.uint8)
    for a, o in ((aL, 0), (aR, 20)):
        if a:
            for i in _PURE_Y[a]:
                pure[o + 2 + i - 1] = 1
                pure[o + 8 + i - 1] = 1
    res = (vec + pure) % 2
    sh = [[], []]
    for j, o in enumerate((0, 20)):
        for i in range(6):
            if res[o + 8 + i] != res[o + 14 + i]:
                return "<not pure twist + shifts>"
            if res[o + 8 + i]:
                sh[j].append(i + 1)
    name = (f"b{aL}" if aL else "") + ((f"`{aR}" if aL else f"b`{aR}") if aR else "")
    s = "".join(map(str, sh[0])) + "".join("`" + str(i) for i in sh[1])
    return (name or "E") + (" + e" + s if s else "")


def fmt_g(p) -> str:
    pi, fl = p
    mv = ", ".join(f"{i+1}->{pi[i]+1}" for i in range(6) if pi[i] != i)
    f = "".join(str(i + 1) for i in range(6) if fl[i])
    return (mv or "identity") + (f" ; y<->w on {f}" if f else "")