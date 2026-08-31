"""
FF_equivalence_checker_master.py

Shared machinery for the classification of order-two T-folds on the SO(12)
lattice at the free fermionic point.

Point-group independent: every routine works on the 40-component boundary
condition vectors and on however many basis rows it is given.  Only the
build_basis_* helpers at the end know about a particular point group.

Fermion layout of the 40-component boundary condition vectors:

  index   0, 1      psi^mu                 |  20,21      psibar^mu
          2 - 7     chi^1..chi^6           |  22 - 27    chibar^1..chibar^6
          8 -13     y^1..y^6               |  28 - 33    ybar^1..ybar^6
         14 -19     w^1..w^6               |  34 - 39    wbar^1..wbar^6

Point groups supported
----------------------
  Z2L_Z2R_Z2   basis {1, S, Sbar, B1, B1b, B2b2}
               parameters n12, N ; k12, K ; m3456      (k = nbar, K = Nbar)
  Z2L_2_Z2R_2  basis {1, S, Sbar, B1, B2, B1b, B2b}
               parameters n12, N, m3456, M ; k12, K, l3456, L
                                                       (k = nbar, l = mbar,
                                                        K = Nbar, L = Mbar)

Equivalence relations (section 4.1)
-----------------------------------
  E1  GL(|B|;Z) changes of basis   -> handled by comparing additive sets Xi
  E2  y^i <-> w^i, per direction, either chirality independently
  E3  permutation of the holomorphic, or of the anti-holomorphic, indices

E2 and E3 generate G = G_L x G_R with |G_L| = |G_R| = 6! * 2^6 = 46080.
find_equivalence() searches all 46080^2 elements without enumerating them:
g_L is screened on the multiset of left halves of Xi, then g_R against the
right halves that the surviving g_L allows.  Every hit is verified explicitly.

Modular invariance (3.4) is imposed on the vectors, using the Minkowskian
product alpha.beta = 1/2 alpha_L.beta_L - 1/2 alpha_R.beta_R:

    beta_a . beta_a = 0 mod 4   <=>  tr_L - tr_R = 0 mod 8
    beta_a . beta_b = 0 mod 2   <=>  ov_L - ov_R = 0 mod 4
    beta_a n beta_b n beta_c n beta_d = 0 mod 1
                                <=>  #{positions where all four are 1} is even
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

POW40 = (1 << np.arange(40, dtype=np.int64))
POW20 = (1 << np.arange(20, dtype=np.int64))

BASIS_NAMES = {
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
    Data about the right half of each element of Xi that no g_R can change:
    the number of periodic fermions and the number of periodic chibar's.
    Attaching it to the left code makes the g_L screening far more selective,
    which matters for configurations with a large symmetry group.
    """
    trR = xi[:, 20:].astype(np.int64).sum(1)
    chiR = xi[:, 22:28].astype(np.int64).sum(1)
    return trR * 8 + chiR


def prep_source(xi: np.ndarray, basis: Optional[np.ndarray] = None) -> Dict:
    """
    Data for the configuration being transformed (the expensive side).
    Passing the basis caches the per-basis-vector code tables, which is what
    makes repeated find_equivalence calls against the same source cheap.
    """
    key = (xi[:, :20].astype(np.int64) @ WH.T) * 256 + _right_invariant(xi)[:, None]
    allL = np.sort(key, axis=0)
    w = _WEIGHTS[:xi.shape[0]]
    out = dict(xi=xi, allL=allL, hash_all=(allL * w[:, None]).sum(0))
    if basis is not None:
        out["basis"] = basis
        out["bl"] = basis[:, :20].astype(np.int64)
        out["rc_all"] = basis[:, 20:].astype(np.int64) @ WH.T
    return out


def prep_target(xi: np.ndarray) -> Dict:
    """Data for the configuration transformed onto (cheap)."""
    lc = xi[:, :20].astype(np.int64) @ POW20
    rc = xi[:, 20:].astype(np.int64) @ POW20
    tmp: Dict[int, List[int]] = {}
    for a, b in zip(lc.tolist(), rc.tolist()):
        tmp.setdefault(a, []).append(b)
    lut = {a: np.array(sorted(set(v)), dtype=np.int64) for a, v in tmp.items()}
    key = np.sort(lc * 256 + _right_invariant(xi))
    w = _WEIGHTS[:xi.shape[0]]
    return dict(xi=xi, sorted_lc=key, lut=lut,
                target=int((key * w).sum()),
                codes=np.sort(encode(xi)))


def find_equivalence(A: Dict, B: Dict, basis_a: Optional[np.ndarray] = None,
                     max_gl: int = 200000):
    """
    Search all of G_L x G_R for g with g(Xi_A) = Xi_B.
    A from prep_source, B from prep_target.  Returns ((piL,flL),(piR,flR)) or None.
    """
    cand = np.flatnonzero(A["hash_all"] == B["target"])
    cand = [c for c in cand if np.array_equal(A["allL"][:, c], B["sorted_lc"])]
    if not cand:
        return None
    bl, rc_all = A.get("bl"), A.get("rc_all")
    if bl is None or rc_all is None:                  # not cached: compute now
        if basis_a is None:
            raise ValueError("pass the basis to prep_source or to find_equivalence")
        bl = basis_a[:, :20].astype(np.int64)
        rc_all = basis_a[:, 20:].astype(np.int64) @ WH.T
    for gl in cand[:max_gl]:
        allowed = [B["lut"].get(int(k)) for k in bl @ WH[gl]]
        if any(a is None for a in allowed):
            continue
        ok = None
        for i in sorted(range(len(allowed)), key=lambda j: len(allowed[j])):
            pos = np.searchsorted(allowed[i], rc_all[i])
            np.clip(pos, 0, len(allowed[i]) - 1, out=pos)
            m = allowed[i][pos] == rc_all[i]
            ok = m if ok is None else (ok & m)
            if not ok.any():
                ok = None
                break
        if ok is None:
            continue
        gL, gR = PARAMS[gl], PARAMS[int(np.flatnonzero(ok)[0])]
        tgt = np.concatenate([_half_targets(*gL), 20 + _half_targets(*gR)])
        img = np.zeros_like(A["xi"])
        img[:, tgt] = A["xi"]                          # vectorised apply_g
        if np.array_equal(np.sort(encode(img)), B["codes"]):
            return gL, gR
    return None


def fingerprint(xi: np.ndarray) -> Tuple:
    """Relabelling invariant used to bucket configurations before testing."""
    trL = xi[:, :20].sum(1)
    trR = xi[:, 20:].sum(1)
    return tuple(sorted(zip(trL.tolist(), trR.tolist())))


# --------------------------------------------------------------------------
# E1: express a vector in a given basis
# --------------------------------------------------------------------------
def solve_F2(basis: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    """Coefficients c with c.basis = target (mod 2), or None."""
    n = basis.shape[0]
    aug = np.concatenate([basis.copy() % 2, np.eye(n, dtype=np.uint8)], axis=1)
    piv, r = [], 0
    for col in range(40):
        p = next((i for i in range(r, n) if aug[i, col]), None)
        if p is None:
            continue
        aug[[r, p]] = aug[[p, r]]
        for i in range(n):
            if i != r and aug[i, col]:
                aug[i] = (aug[i] + aug[r]) % 2
        piv.append(col); r += 1
        if r == n:
            break
    t = target.copy() % 2
    coef = np.zeros(n, dtype=np.uint8)
    for i, col in enumerate(piv):
        if t[col]:
            t = (t + aug[i, :40]) % 2
            coef = (coef + aug[i, 40:]) % 2
    return None if t.any() else coef


def basis_change_note(src_twists, src_names, dst_basis, dst_names, g) -> str:
    """
    How the E2/E3 images of the source twist basis vectors sit inside the
    target basis.  '1 + S + Sb' is abbreviated to E.
    """
    parts = []
    for name, vec in zip(src_names, src_twists):
        coef = solve_F2(dst_basis, apply_g(vec, g[0], g[1]))
        if coef is None:
            return "?"
        terms = [dst_names[j] for j in range(len(coef)) if coef[j]]
        if all(x in terms for x in ("1", "S", "Sb")):
            terms = ["E"] + [x for x in terms if x not in ("1", "S", "Sb")]
        parts.append(f"{name} -> " + (" + ".join(terms) if terms else "0"))
    return "; ".join(parts)


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------
_PURE = {frozenset(): None, frozenset({3, 4, 5, 6}): "1",
         frozenset({1, 2, 5, 6}): "2", frozenset({1, 2, 3, 4}): "3"}
_PURE_Y = {"1": {3, 4, 5, 6}, "2": {1, 2, 5, 6}, "3": {1, 2, 3, 4}}


def describe(vec: np.ndarray) -> str:
    """'b1 + e12`3`4' -- a backtick marks an anti-holomorphic index."""
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


def equivalence_note(src, dst, g, names) -> str:
    """The human-readable note written into the output csv."""
    return (f"duplicate of {dst['label']}. "
            f"E3/E2 permutations -- holomorphic: {fmt_g(g[0])}; "
            f"anti-holomorphic: {fmt_g(g[1])}. "
            f"E1 change of basis (E = 1+S+Sb): "
            f"{basis_change_note(src['twists'], names[3:], dst['basis'], names, g)}")
