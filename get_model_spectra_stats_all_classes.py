#!/usr/bin/env python3
"""
get_model_spectra_stats_all_classes.py

Unified parallel driver for all nine point-group classes:
  Z2, Z2_2, Z2L,
  Z2L_2, Z2L_Z2R, Z2L_2_Z2R, Z2L_2_Z2R_2, Z2L_Z2, Z2L_Z2R_Z2

Uses TypeIIFreeFermioniser_v5.py.

Basis structures
----------------
  Z2          : ['1', S, Sbar, B_{1b1}]                   symmetric Z2 (chi3..6 base)
  Z2_2        : ['1', S, Sbar, B_{1b1}, B_{2b2}]          two symmetric Z2 twists
                  B_{1b1} = make_B1b1(n1,n2)  chi3..6 base
                  B_{2b2} = make_B2b2([m3,m4,m5,m6])  chi1,2,5,6 base
  Z2L         : ['1', S, Sbar, B1]                        left-only Z2L
  Z2L_2       : ['1', S, Sbar, B1, B2]
  Z2L_Z2R     : ['1', S, Sbar, B1, B_b1]
  Z2L_2_Z2R   : ['1', S, Sbar, B1, B2, B_b1]
  Z2L_2_Z2R_2 : ['1', S, Sbar, B1, B2, B_b1, B_b2]
  Z2L_Z2      : ['1', S, Sbar, B1, B_{2b2}]
  Z2L_Z2R_Z2  : ['1', S, Sbar, B1, B_b1, B_{2b2}]

Phase scanning
--------------
  IIA/IIB (top-level, every class, both always scanned):
    Sets C(1,1) (=+1 IIB, -1 IIA) and C(1,Sbar) (same sign) in the base 7x7
    GGSO template before any structural/SUSY overrides are applied.
    C(Sbar,Sbar) is not independently free -- modular invariance forces
    C(Sbar,Sbar) = -C(1,Sbar) regardless of the template's own (Sbar,Sbar)
    entry, so fixing C(1,1)/C(1,Sbar) is equivalent to fixing C(Sbar,Sbar).

  Structural variant phases (per class, scanned within each of IIA/IIB):
    Z2_2        : C_B1b1_B2b2                              -> 2 combos
    Z2L_Z2R     : C_B1_Bb1                                  -> 2 combos
    Z2L_2_Z2R   : C_B1_Bb1, C_B2_Bb1                       -> 4 combos
    Z2L_Z2R_Z2  : C_B1_Bb1, C_B1_B2b2, C_Bb1_B2b2         -> 8 combos
    Z2L_2_Z2R_2 : C_B1_Bb1, C_B1_Bb2, C_B2_Bb1, C_B2_Bb2 -> 16 combos
    others      : none (single structural run)

  SUSY-breaking phases (auto-detected per class):
    Z2, Z2_2    : none (all twists symmetric)
    Z2L         : C(Sbar,B1)                      -> +1 breaking variant
    Z2L_2       : C(Sbar,B1), C(Sbar,B2)          -> 3 breaking variants
    Z2L_Z2R     : C(Sbar,B1), C(S,B_b1)           -> 3 breaking variants
    Z2L_2_Z2R   : C(Sbar,B1), C(Sbar,B2),
                  C(S,B_b1)                        -> 7 breaking variants
    Z2L_2_Z2R_2 : C(Sbar,B1), C(Sbar,B2),
                  C(S,B_b1), C(S,B_b2)            -> 15 breaking variants
    Z2L_Z2      : C(Sbar,B1)                       -> +1 breaking variant
    Z2L_Z2R_Z2  : C(Sbar,B1), C(S,B_b1)           -> 3 breaking variants

Outputs
-------
  Spectra_Stats_ByClass_SUSYBROKEN/{source}_spectra_stats_SUSY.csv
  Spectra_Stats_ByClass_SUSYBROKEN/{source}_spectra_stats_non_SUSY.csv
  Spectra_Stats_ByClass_SUSYBROKEN/{source}_spectra_stats_SUSY_collapsed.csv
  Spectra_Stats_ByClass_SUSYBROKEN/{source}_spectra_stats_non_SUSY_collapsed.csv
  Processed_Spectra_SUSY/{source}/{IIA,IIB}/{run_label}_processed.csv
  Processed_Spectra_non_SUSY/{source}/{IIA,IIB}/{run_label}_processed.csv

  Per-point-group/-type-II subfolders ({source}/{IIA,IIB}) are created lazily,
  so only point groups currently enabled in INPUT_FILES get folders. The
  IIA/IIB choice is also recorded as a TYPE_II column in the stats CSVs.
"""

import ast
import csv
import importlib.util
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

TYPEIIFF_PATH = os.path.join(HERE, "TypeIIFreeFermioniser_v5.py")
MULT_CSV_PATH = os.path.join(HERE, "Input_typeII", "supermultiplets.csv")

_INPUT_DIR = os.path.join(HERE, "All_Z2N_Input_Models_updated_310526")

INPUT_FILES: Dict[str, str] = {
    #"Z2":          os.path.join(_INPUT_DIR, "Z2_solutions.csv"),
    #"Z2_2":        os.path.join(_INPUT_DIR, "Z2_2_solutions.csv"),
    #"Z2L":         os.path.join(_INPUT_DIR, "Z2L_solutions.csv"),
    #"Z2L_2":       os.path.join(_INPUT_DIR, "Z2L_2_stefan_solutions.csv"),
    #"Z2L_Z2R":     os.path.join(_INPUT_DIR, "Z2L_Z2R_stefan_solutions.csv"),
    #"Z2L_Z2":      os.path.join(_INPUT_DIR, "Z2L_Z2_stefan_solutions.csv"),
    #"Z2L_2_Z2R":   os.path.join(_INPUT_DIR, "Z2L_2_Z2R_v2_reformatted.csv"),
    "Z2L_2_Z2R_2": os.path.join(_INPUT_DIR, "Z2L_2_Z2R_2_v2_reformatted.csv"),
    #"Z2L_Z2R_Z2":  os.path.join(_INPUT_DIR, "Z2L_Z2R_Z2_v2_reformatted.csv"),
}

OUT_DIR              = os.path.join(HERE, "Spectra_Stats_ByClass_SUSYBROKEN")
PROCESSED_DIR_SUSY   = os.path.join(HERE, "Processed_Spectra_SUSY")
PROCESSED_DIR_NON_SUSY = os.path.join(HERE, "Processed_Spectra_non_SUSY")
os.makedirs(OUT_DIR,                exist_ok=True)
os.makedirs(PROCESSED_DIR_SUSY,     exist_ok=True)
os.makedirs(PROCESSED_DIR_NON_SUSY, exist_ok=True)

WRITE_PROCESSED   = True
PRINT_EACH_RESULT = True
SCAN_NON_SUSY     = False  # set True to also scan SUSY-breaking (non_SUSY) variants
MAX_N_SUSY        = 2   # filter threshold for non_SUSY jobs

_cpu = os.cpu_count() or 2
MAX_WORKERS = (
    min(61, max(1, _cpu - 1)) if os.name == "nt"
    else max(1, _cpu - 1)
)


# ── Shared arithmetic helpers ─────────────────────────────────────────────────

def _as_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    s = str(x).strip()
    return default if s == "" else (int(s) if s.lstrip("-").isdigit() else default)


def _split_lr(vec40: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return vec40[:20], vec40[20:]


def _comp_dim_from_basis(basis: np.ndarray) -> int:
    return (basis.shape[1] - 16) // 4


def _dot_prod(b1: np.ndarray, b2: np.ndarray, comp_dim: int) -> float:
    cut = 8 + comp_dim * 2
    end = 16 + comp_dim * 4
    return float(np.dot(b1[:cut], b2[:cut]) - np.dot(b1[cut:end], b2[cut:end]))


def _snap_pm1(val: complex) -> int:
    return 1 if np.real(val) >= 0 else -1


def _parse_tuple_cell(x: Any) -> Tuple[int, ...]:
    if isinstance(x, (tuple, list, np.ndarray)):
        return tuple(int(v) for v in x)
    s = str(x).strip()
    if not s:
        return tuple()
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (tuple, list)):
            return tuple(int(v) for v in obj)
    except Exception:
        pass
    s = s.strip("()[]")
    parts = [p for p in s.replace(",", " ").split() if p]
    return tuple(int(p) for p in parts)


def _parse_match_str_to_dict(match_str: str, mult_names: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in mult_names}
    if not match_str or match_str.startswith("no ") or match_str.startswith("0 "):
        return counts
    for part in match_str.split("+"):
        part = part.strip()
        if not part or "*" not in part:
            continue
        try:
            c_str, name = part.split("*", 1)
            name = name.strip()
            if name in counts:
                counts[name] = int(c_str.strip())
        except Exception:
            continue
    return counts


def _format_supersector_keys(keys: Any) -> str:
    if keys is None:
        return ""
    if isinstance(keys, str):
        return keys
    try:
        formatted: List[str] = []
        for key in keys:
            key_t = tuple(int(v) for v in key)
            display = [str(key_t[0]), "*", "*"] + [str(x) for x in key_t[1:]]
            formatted.append("[" + ", ".join(display) + "]")
        return "[" + ", ".join(formatted) + "]"
    except Exception:
        return str(keys)


def read_table_auto(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    return df


# ── Fixed basis vectors b1..b7 ────────────────────────────────────────────────

def make_one() -> np.ndarray:
    return np.ones(40, dtype=int)


def make_S() -> np.ndarray:
    v = np.zeros(40, dtype=int); v[:8] = 1; return v


def make_Sbar() -> np.ndarray:
    v = np.zeros(40, dtype=int); v[20:28] = 1; return v


def make_B1(n1: int, n2: int, N: List[int]) -> np.ndarray:
    v = np.zeros(40, dtype=int)
    L, R = _split_lr(v)
    L[4:8] = 1
    L[8:10] = [n1, n2]
    L[10:14] = 1
    L[14:16] = [n1, n2]
    R[8:14] = N
    R[14:20] = N
    return v


def make_B2(m: List[int], M: List[int]) -> np.ndarray:
    m3, m4, m5, m6 = m
    v = np.zeros(40, dtype=int)
    L, R = _split_lr(v)
    L[2:4] = 1
    L[6:8] = 1
    L[8:10] = [1, 1]
    L[10:12] = [m3, m4]
    L[12:14] = [1 - m5, 1 - m6]
    L[16:18] = [m3, m4]
    L[18:20] = [m5, m6]
    R[8:14] = M
    R[14:20] = M
    return v


def make_Bb1(k1: int, k2: int, K: List[int]) -> np.ndarray:
    v = np.zeros(40, dtype=int)
    L, R = _split_lr(v)
    L[8:14] = K
    L[14:20] = K
    R[4:8] = 1
    R[8:10] = [k1, k2]
    R[10:14] = 1
    R[14:16] = [k1, k2]
    return v


def make_Bb2(l: List[int], L6: List[int]) -> np.ndarray:
    l3, l4, l5, l6 = l
    v = np.zeros(40, dtype=int)
    Lh, Rh = _split_lr(v)
    Lh[8:14] = L6
    Lh[14:20] = L6
    Rh[2:4] = 1
    Rh[6:8] = 1
    Rh[8:10] = [1, 1]
    Rh[10:12] = [l3, l4]
    Rh[12:14] = [1 - l5, 1 - l6]
    Rh[16:18] = [l3, l4]
    Rh[18:20] = [l5, l6]
    return v


def make_B2b2(m: List[int], mb: Optional[List[int]] = None) -> np.ndarray:
    if mb is None:
        mb = m
    m3, m4, m5, m6 = m
    mb3, mb4, mb5, mb6 = mb
    v = np.zeros(40, dtype=int)
    L, R = _split_lr(v)
    L[2:4] = 1
    L[6:8] = 1
    L[8:14]  = [1, 1, m3,  m4,  1 - m5,  1 - m6]
    L[14:20] = [0, 0, m3,  m4,  m5,      m6]
    R[2:4] = 1
    R[6:8] = 1
    R[8:14]  = [1, 1, mb3, mb4, 1 - mb5, 1 - mb6]
    R[14:20] = [0, 0, mb3, mb4, mb5,     mb6]
    return v


def make_B1b1(n1: int, n2: int) -> np.ndarray:
    """Symmetric B_{1bar1} twist: chi3..6 base on both sides."""
    v = np.zeros(40, dtype=int)
    v[4:8]   = 1      # left  chi3..chi6
    v[8]     = n1;  v[9]     = n2
    v[10:14] = 1
    v[14]    = n1;  v[15]    = n2
    v[24:28] = 1      # right chi3..chi6
    v[28]    = n1;  v[29]    = n2
    v[30:34] = 1
    v[34]    = n1;  v[35]    = n2
    return v


# ── Basis builders per class ──────────────────────────────────────────────────

def build_basis_Z2(row: pd.Series) -> np.ndarray:
    m1 = _as_int(row.get("m1")); m2 = _as_int(row.get("m2"))
    return np.vstack([make_one(), make_S(), make_Sbar(), make_B1b1(m1, m2)]).astype(int)


def build_basis_Z2_2(row: pd.Series) -> np.ndarray:
    n1 = _as_int(row.get("n1")); n2 = _as_int(row.get("n2"))
    m3 = _as_int(row.get("m3")); m4 = _as_int(row.get("m4"))
    m5 = _as_int(row.get("m5")); m6 = _as_int(row.get("m6"))
    return np.vstack([
        make_one(), make_S(), make_Sbar(),
        make_B1b1(n1, n2),
        make_B2b2([m3, m4, m5, m6]),
    ]).astype(int)


def build_basis_Z2L(row: pd.Series) -> np.ndarray:
    n1 = _as_int(row.get("n1")); n2 = _as_int(row.get("n2"))
    N = [_as_int(row.get(f"N{i}")) for i in range(1, 7)]
    return np.vstack([make_one(), make_S(), make_Sbar(), make_B1(n1, n2, N)]).astype(int)


def build_basis_Z2L_2(row: pd.Series) -> np.ndarray:
    n1 = _as_int(row.get("n1")); n2 = _as_int(row.get("n2"))
    N = [_as_int(row.get(f"N{i}")) for i in range(1, 7)]
    m = [_as_int(row.get(f"m{i}")) for i in range(3, 7)]
    M = [_as_int(row.get(f"M{i}")) for i in range(1, 7)]
    return np.vstack([make_one(), make_S(), make_Sbar(), make_B1(n1, n2, N), make_B2(m, M)]).astype(int)


def build_basis_Z2L_Z2R(row: pd.Series) -> np.ndarray:
    n1 = _as_int(row.get("n1")); n2 = _as_int(row.get("n2"))
    N = [_as_int(row.get(f"N{i}")) for i in range(1, 7)]
    k1 = _as_int(row.get("k1")); k2 = _as_int(row.get("k2"))
    K = [_as_int(row.get(f"K{i}")) for i in range(1, 7)]
    return np.vstack([make_one(), make_S(), make_Sbar(), make_B1(n1, n2, N), make_Bb1(k1, k2, K)]).astype(int)


def build_basis_Z2L_2_Z2R(row: pd.Series) -> np.ndarray:
    n12 = _parse_tuple_cell(row.get("n12"))
    m   = _parse_tuple_cell(row.get("m3456"))
    k12 = _parse_tuple_cell(row.get("k12"))
    N   = _parse_tuple_cell(row.get("N"))
    M   = _parse_tuple_cell(row.get("M"))
    K   = _parse_tuple_cell(row.get("K"))
    if len(n12) != 2 or len(k12) != 2 or len(m) != 4 or len(N) != 6 or len(M) != 6 or len(K) != 6:
        raise ValueError("Bad tuple lengths in Z2L_2_Z2R row")
    return np.vstack([
        make_one(), make_S(), make_Sbar(),
        make_B1(n12[0], n12[1], list(N)),
        make_B2(list(m), list(M)),
        make_Bb1(k12[0], k12[1], list(K)),
    ]).astype(int)


def build_basis_Z2L_2_Z2R_2(row: pd.Series) -> np.ndarray:
    n12 = _parse_tuple_cell(row.get("n"))
    m   = _parse_tuple_cell(row.get("m"))
    k12 = _parse_tuple_cell(row.get("k"))
    l   = _parse_tuple_cell(row.get("l"))
    N   = _parse_tuple_cell(row.get("N"))
    M   = _parse_tuple_cell(row.get("M"))
    K   = _parse_tuple_cell(row.get("K"))
    L6  = _parse_tuple_cell(row.get("L"))
    if len(n12) != 2 or len(k12) != 2 or len(m) != 4 or len(l) != 4:
        raise ValueError("Bad tuple lengths in Z2L_2_Z2R_2 row (n/m/k/l)")
    if len(N) != 6 or len(M) != 6 or len(K) != 6 or len(L6) != 6:
        raise ValueError("Bad tuple lengths in Z2L_2_Z2R_2 row (N/M/K/L)")
    return np.vstack([
        make_one(), make_S(), make_Sbar(),
        make_B1(n12[0], n12[1], list(N)),
        make_B2(list(m), list(M)),
        make_Bb1(k12[0], k12[1], list(K)),
        make_Bb2(list(l), list(L6)),
    ]).astype(int)


def build_basis_Z2L_Z2(row: pd.Series) -> np.ndarray:
    n1 = _as_int(row.get("n1")); n2 = _as_int(row.get("n2"))
    N = [_as_int(row.get(f"N{i}")) for i in range(1, 7)]
    m = [_as_int(row.get(f"m{i}")) for i in range(3, 7)]
    return np.vstack([make_one(), make_S(), make_Sbar(), make_B1(n1, n2, N), make_B2b2(m, m)]).astype(int)


def build_basis_Z2L_Z2R_Z2(row: pd.Series) -> np.ndarray:
    n1 = _as_int(row.get("n1")); n2 = _as_int(row.get("n2"))
    N  = [_as_int(row.get(f"N{i}")) for i in range(1, 7)]
    k1 = _as_int(row.get("k1")); k2 = _as_int(row.get("k2"))
    K  = [_as_int(row.get(f"K{i}")) for i in range(1, 7)]
    m  = [_as_int(row.get(f"m{i}")) for i in range(3, 7)]
    return np.vstack([
        make_one(), make_S(), make_Sbar(),
        make_B1(n1, n2, N), make_Bb1(k1, k2, K), make_B2b2(m, m),
    ]).astype(int)


BASIS_BUILDERS: Dict[str, Any] = {
    "Z2":          build_basis_Z2,
    "Z2_2":        build_basis_Z2_2,
    "Z2L":         build_basis_Z2L,
    "Z2L_2":       build_basis_Z2L_2,
    "Z2L_Z2R":     build_basis_Z2L_Z2R,
    "Z2L_2_Z2R":   build_basis_Z2L_2_Z2R,
    "Z2L_2_Z2R_2": build_basis_Z2L_2_Z2R_2,
    "Z2L_Z2":      build_basis_Z2L_Z2,
    "Z2L_Z2R_Z2":  build_basis_Z2L_Z2R_Z2,
}


# ── GGSO template and enforced fill ──────────────────────────────────────────

# Row/col meaning: 0='1', 1=S, 2=Sbar, 3=B1/B_{1b1}, 4=B2/B_{2b2}, 5=B_b1, 6=B_b2
# T[1,2]=C(S,Sbar)=+1 mandatory for gravitinos (same for IIA and IIB).
#
# IIA/IIB convention: C(1,1)=+1 for IIB, -1 for IIA (T[0,0]). C(1,Sbar) follows
# the same sign (T[0,2]). Note C(Sbar,Sbar) is NOT independently free: the
# diagonal-fill step in gso_from_template derives it as C(Sbar,Sbar) =
# -C(1,Sbar) automatically via modular invariance, so T[2,2] here is set to
# match for clarity but is overwritten regardless of its literal value.
def _template_7x7(type_ii: str) -> np.ndarray:
    if type_ii not in ("IIA", "IIB"):
        raise ValueError(f"type_ii must be 'IIA' or 'IIB', got {type_ii!r}")
    s = +1 if type_ii == "IIB" else -1
    return np.array(
        [
            [ s, +1,  s, -1, -1, -1, -1],
            [ 0, -1, +1, -1, -1, -1, -1],
            [ 0,  0, -s, -1, -1, -1, -1],
            [ 0,  0,  0, +1, +1, +1, +1],
            [ 0,  0,  0,  0, +1, +1, +1],
            [ 0,  0,  0,  0,  0, +1, +1],
            [ 0,  0,  0,  0,  0,  0, +1],
        ],
        dtype=int,
    )

# For each class, which rows/cols of the 7×7 template to use (in basis order).
TEMPLATE_INDEX_MAP: Dict[str, List[int]] = {
    "Z2":          [0, 1, 2, 3],
    "Z2_2":        [0, 1, 2, 3, 4],
    "Z2L":         [0, 1, 2, 3],
    "Z2L_2":       [0, 1, 2, 3, 4],
    "Z2L_Z2R":     [0, 1, 2, 3, 5],
    "Z2L_2_Z2R":   [0, 1, 2, 3, 4, 5],
    "Z2L_2_Z2R_2": [0, 1, 2, 3, 4, 5, 6],
    "Z2L_Z2":      [0, 1, 2, 3, 4],
    "Z2L_Z2R_Z2":  [0, 1, 2, 3, 5, 4],
}


def gso_from_template(
    basis: np.ndarray,
    source: str,
    type_ii: str,
    upper_overrides: Optional[Dict[Tuple[int, int], int]] = None,
) -> np.ndarray:
    """Build the GGSO matrix from the maximal-SUSY template with optional
    upper-triangle overrides.  Lower triangle and diagonal are filled by MI."""
    if source not in TEMPLATE_INDEX_MAP:
        raise ValueError(f"Unknown source: {source!r}")
    idx = TEMPLATE_INDEX_MAP[source]
    Ts  = _template_7x7(type_ii)[np.ix_(idx, idx)]
    n   = Ts.shape[0]
    comp_dim = _comp_dim_from_basis(basis)

    G = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            val = Ts[i, j]
            G[i, j] = 1 if val == 0 else int(np.sign(val))

    if upper_overrides:
        for (i, j), sgn in upper_overrides.items():
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if not (0 <= a < n and 0 <= b < n):
                raise IndexError(f"Override ({i},{j}) out of range for n={n}")
            if sgn not in (-1, 1):
                raise ValueError(f"Override sign must be ±1, got {sgn}")
            G[a, b] = int(sgn)

    for i in range(n):
        for j in range(i + 1, n):
            n_ij  = _dot_prod(basis[i], basis[j], comp_dim)
            phase = np.exp(1j * np.pi * n_ij / 4.0)
            G[j, i] = _snap_pm1(phase * np.conj(G[i, j]))

    for i in range(n):
        n_ii   = _dot_prod(basis[i], basis[i], comp_dim)
        phase2 = np.exp(1j * np.pi * n_ii / 8.0)
        G[i, i] = _snap_pm1(phase2 * np.conj(G[i, 0]))

    return G


# ── Structural (non-SUSY-breaking) variant phases ─────────────────────────────

def structural_variants(source: str) -> List[Dict[str, int]]:
    """Phase combinations that select V/H content, NOT SUSY-breaking.

    IIA/IIB is handled as a top-level loop in main() (it sets C(1,1)/C(1,Sbar)
    in the base GGSO template directly), so it no longer appears here."""
    if source == "Z2_2":
        return [{"C_B1b1_B2b2": b} for b in (+1, -1)]
    if source == "Z2L_Z2R":
        return [{"C_B1_Bb1": a} for a in (-1, +1)]
    if source == "Z2L_2_Z2R":
        return [
            {"C_B1_Bb1": a, "C_B2_Bb1": b}
            for a in (-1, +1) for b in (-1, +1)
        ]
    if source == "Z2L_Z2R_Z2":
        return [
            {"C_B1_Bb1": a, "C_B1_B2b2": b, "C_Bb1_B2b2": c}
            for a in (-1, +1) for b in (-1, +1) for c in (-1, +1)
        ]
    if source == "Z2L_2_Z2R_2":
        return [
            {"C_B1_Bb1": a, "C_B1_Bb2": b, "C_B2_Bb1": c, "C_B2_Bb2": d}
            for a in (-1, +1) for b in (-1, +1) for c in (-1, +1) for d in (-1, +1)
        ]
    return [{}]


def structural_overrides(
    source: str, choice: Dict[str, int],
) -> Dict[Tuple[int, int], int]:
    """Translate structural phase choice to upper-triangle GGSO override indices."""
    if source == "Z2_2":
        out: Dict[Tuple[int, int], int] = {}
        if "C_B1b1_B2b2" in choice:
            out[(3, 4)] = choice["C_B1b1_B2b2"]
        return out
    if source == "Z2L_Z2R":
        return {(3, 4): choice["C_B1_Bb1"]}
    if source == "Z2L_2_Z2R":
        return {(3, 5): choice["C_B1_Bb1"], (4, 5): choice["C_B2_Bb1"]}
    if source == "Z2L_Z2R_Z2":
        return {(3, 4): choice["C_B1_Bb1"], (3, 5): choice["C_B1_B2b2"], (4, 5): choice["C_Bb1_B2b2"]}
    if source == "Z2L_2_Z2R_2":
        return {(3, 5): choice["C_B1_Bb1"], (3, 6): choice["C_B1_Bb2"],
                (4, 5): choice["C_B2_Bb1"], (4, 6): choice["C_B2_Bb2"]}
    return {}


def structural_phase_cols(source: str) -> List[str]:
    if source == "Z2_2":        return ["C_B1b1_B2b2"]
    if source == "Z2L_Z2R":     return ["C_B1_Bb1"]
    if source == "Z2L_2_Z2R":   return ["C_B1_Bb1", "C_B2_Bb1"]
    if source == "Z2L_Z2R_Z2":  return ["C_B1_Bb1", "C_B1_B2b2", "C_Bb1_B2b2"]
    if source == "Z2L_2_Z2R_2": return ["C_B1_Bb1", "C_B1_Bb2", "C_B2_Bb1", "C_B2_Bb2"]
    return []


def structural_phase_tag(source: str, choice: Dict[str, int]) -> str:
    if not choice:
        return "novar"
    cols = structural_phase_cols(source)
    return "_".join("p" if choice.get(c, +1) == +1 else "m" for c in cols)


# ── SUSY-breaking phase infrastructure ────────────────────────────────────────

def _overlap_S(b: np.ndarray) -> int:
    return int(np.sum(b[0:8] == 1))


def _overlap_Sbar(b: np.ndarray) -> int:
    return int(np.sum(b[20:28] == 1))


def susy_breaking_indices(basis: np.ndarray) -> Dict[str, List[int]]:
    """Return indices i>=3 where C(S,bi) or C(Sbar,bi) is a free SUSY-breaking knob."""
    n = basis.shape[0]
    left_idx:  List[int] = []
    right_idx: List[int] = []
    for i in range(3, n):
        if _overlap_S(basis[i])    == 0: left_idx.append(i)
        if _overlap_Sbar(basis[i]) == 0: right_idx.append(i)
    return {"left": left_idx, "right": right_idx}


def susy_preserving_choice(left_idx: List[int], right_idx: List[int]) -> Dict[str, int]:
    choice: Dict[str, int] = {}
    for i in left_idx:  choice[f"S_{i}"]    = -1
    for i in right_idx: choice[f"Sbar_{i}"] = -1
    return choice


def susy_breaking_choices(left_idx: List[int], right_idx: List[int]) -> List[Dict[str, int]]:
    """All non-trivial SUSY-breaking sign assignments (excluding all-preserve)."""
    out: List[Dict[str, int]] = []
    nL, nR = len(left_idx), len(right_idx)
    for mask in range(1, 1 << (nL + nR)):
        choice: Dict[str, int] = {}
        for k, idx in enumerate(left_idx):
            choice[f"S_{idx}"]    = +1 if (mask >> k) & 1 else -1
        for k, idx in enumerate(right_idx):
            choice[f"Sbar_{idx}"] = +1 if (mask >> (nL + k)) & 1 else -1
        out.append(choice)
    return out


def susy_choice_to_overrides(susy_choice: Dict[str, int]) -> Dict[Tuple[int, int], int]:
    """S is row 1, Sbar is row 2 in the basis; i >= 3."""
    out: Dict[Tuple[int, int], int] = {}
    for key, sgn in susy_choice.items():
        if key.startswith("S_"):
            out[(1, int(key[2:]))] = int(sgn)
        elif key.startswith("Sbar_"):
            out[(2, int(key[5:]))] = int(sgn)
    return out


def format_susy_tag(susy_choice: Dict[str, int]) -> str:
    S_parts    = sorted(((int(k[2:]),  v) for k, v in susy_choice.items() if k.startswith("S_")),    key=lambda x: x[0])
    Sbar_parts = sorted(((int(k[5:]),  v) for k, v in susy_choice.items() if k.startswith("Sbar_")), key=lambda x: x[0])
    s_str    = "".join("p" if v == +1 else "m" for _, v in S_parts)
    sbar_str = "".join("p" if v == +1 else "m" for _, v in Sbar_parts)
    return f"S{s_str}_Sbar{sbar_str}"


def susy_phase_cols(susy_choice: Dict[str, int]) -> List[str]:
    S_keys    = sorted((k for k in susy_choice if k.startswith("S_")),    key=lambda s: int(s.split("_")[1]))
    Sbar_keys = sorted((k for k in susy_choice if k.startswith("Sbar_")), key=lambda s: int(s.split("_")[1]))
    return S_keys + Sbar_keys


# ── Worker import helper ───────────────────────────────────────────────────────

def _import_freefermion(typeiiff_path: str):
    existing = sys.modules.get("TypeIIFreeFermioniser")
    if existing is not None and hasattr(existing, "FreeFermionModel"):
        return existing.FreeFermionModel
    spec = importlib.util.spec_from_file_location("TypeIIFreeFermioniser", typeiiff_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for: {typeiiff_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "FreeFermionModel"):
        raise ImportError("FreeFermionModel not found in loaded module.")
    return mod.FreeFermionModel


# ── Stats columns ─────────────────────────────────────────────────────────────

_STATS_COLS = [
    "TYPE_II",
    "MI_OK",
    "N_SUSY_L", "N_SUSY_R", "N_SUSY",
    "N_RS_L",   "N_RS_R",   "N_RS",
    "V_RR", "H_RR", "V_T", "H_T", "N_V", "N_H",
    "SPIN0", "SPIN_HALF", "SPIN1", "SPIN_3HALF", "SPIN2",
    "INTERNAL_SYMMETRY",
    "TWISTED_SUPERSECTORS", "ASYM_TWIST_SUPERSECTORS", "SYM_TWIST_SUPERSECTORS",
    "MULTIPLET_MATCH",
]
_STR_STATS = {
    "TYPE_II", "MI_OK", "INTERNAL_SYMMETRY",
    "TWISTED_SUPERSECTORS", "ASYM_TWIST_SUPERSECTORS", "SYM_TWIST_SUPERSECTORS",
    "MULTIPLET_MATCH",
}


def _blank_result(
    source, model_i, run_label, type_ii,
    susy_choice, struct_choice, job_type,
    row_dict, err, dt,
) -> Dict[str, Any]:
    return {
        "source": source, "model_i": model_i, "run_label": run_label,
        "TYPE_II": type_ii,
        "job_type": job_type,
        **(susy_choice or {}), **(struct_choice or {}), **row_dict,
        "MI_OK": "FAIL",
        "N_SUSY_L": 0, "N_SUSY_R": 0, "N_SUSY": 0,
        "N_RS_L": 0, "N_RS_R": 0, "N_RS": 0,
        "V_RR": 0, "H_RR": 0, "V_T": 0, "H_T": 0, "N_V": 0, "N_H": 0,
        "SPIN0": 0, "SPIN_HALF": 0, "SPIN1": 0, "SPIN_3HALF": 0, "SPIN2": 0,
        "INTERNAL_SYMMETRY": "",
        "TWISTED_SUPERSECTORS": "", "ASYM_TWIST_SUPERSECTORS": "",
        "SYM_TWIST_SUPERSECTORS": "", "MULTIPLET_MATCH": "",
        "err": err, "dt": dt,
    }


# ── Single-job worker ─────────────────────────────────────────────────────────

def _run_one_job(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t0 = time.time()

    source:        str            = job["source"]
    model_i:       int            = int(job["model_i"])
    type_ii:       str            = job["type_ii"]
    row_dict:      Dict           = job["row_dict"]
    susy_choice:   Dict[str, int] = job["susy_choice"]
    struct_choice: Dict[str, int] = job["struct_choice"]
    job_type:      str            = job.get("job_type", "non_SUSY")
    filter_susy:   bool           = bool(job.get("filter_susy", True))
    typeiiff_path: str            = job["typeiiff_path"]
    mult_csv:      str            = job["mult_csv"]
    write_proc:    bool           = bool(job["write_processed"])
    proc_dir:      str            = job["proc_dir"]
    max_n_susy:    int            = int(job["max_n_susy"])

    # susy_choice is empty exactly when the model class has no free SUSY-breaking
    # knobs at all (e.g. Z2/Z2_2: their symmetric twists only fix helicities of
    # S/Sbar states and cannot break SUSY) — omit the S/Sbar tag in that case
    # rather than printing a misleading placeholder.
    susy_tag   = format_susy_tag(susy_choice) if susy_choice else ""
    struct_tag = structural_phase_tag(source, struct_choice)
    label_parts = [source, type_ii, f"m{model_i}"]
    if susy_tag:
        label_parts.append(susy_tag)
    label_parts.append(struct_tag)
    run_label = "__".join(label_parts)

    try:
        basis = BASIS_BUILDERS[source](pd.Series(row_dict))
    except Exception as e:
        return _blank_result(source, model_i, run_label, type_ii, susy_choice, struct_choice,
                             job_type, row_dict, f"Basis build failed: {e}", time.time() - t0)

    try:
        overrides: Dict[Tuple[int, int], int] = {}
        overrides.update(susy_choice_to_overrides(susy_choice))
        overrides.update(structural_overrides(source, struct_choice))
        gso = gso_from_template(basis, source, type_ii, upper_overrides=overrides)
    except Exception as e:
        return _blank_result(source, model_i, run_label, type_ii, susy_choice, struct_choice,
                             job_type, row_dict, f"GGSO build failed: {e}", time.time() - t0)

    try:
        FreeFermionModel = _import_freefermion(typeiiff_path)
        ff = FreeFermionModel.from_arrays(basis=basis, gso=gso, label=run_label, type_ii=type_ii)
        df_raw, df_proc, stats = ff.compute()

        if filter_susy and stats.n_susy > max_n_susy:
            return None  # filtered out

        if write_proc:
            os.makedirs(proc_dir, exist_ok=True)
            proc_path = os.path.join(proc_dir, f"{run_label}_processed.csv")
            ff.write_processed_csv(
                df_proc, proc_path,
                stats.twisted_supersectors,
                stats.rs_supersectors,
                stats.vh_t_supersectors,
            )

        try:
            sym_info     = ff.compute_internal_symmetry_groups()
            internal_sym = sym_info.get("sym_str", "")
        except Exception:
            internal_sym = ""

        mult_match_str = ""
        mult_counts: Dict[str, int] = {}
        n_susy_val = int(stats.n_susy)
        if n_susy_val >= 1:
            mod = sys.modules.get("TypeIIFreeFermioniser")
            if mod is not None:
                multiplets = None
                if mult_csv:
                    multiplets = mod.load_multiplet_library_csv(mult_csv, n_susy_val)
                if multiplets is None:
                    multiplets = mod._multiplet_library(n_susy_val)
                if multiplets:
                    spin_list = [
                        int(stats.spin_counts_total.get("0",   0)),
                        int(stats.spin_counts_total.get("1/2", 0)),
                        int(stats.spin_counts_total.get("1",   0)),
                        int(stats.spin_counts_total.get("3/2", 0)),
                        int(stats.spin_counts_total.get("2",   0)),
                    ]
                    mult_match_str = mod._match_multiplets_str(spin_list, multiplets)
                    mult_counts = _parse_match_str_to_dict(
                        mult_match_str, [m[0] for m in multiplets]
                    )

        return {
            "source": source, "model_i": model_i, "run_label": run_label,
            "TYPE_II": type_ii,
            "job_type": job_type,
            **susy_choice, **struct_choice, **row_dict,
            "MI_OK":      "OK" if stats.mi_ok else "FAIL",
            "N_SUSY_L":   int(stats.n_susy_L),
            "N_SUSY_R":   int(stats.n_susy_R),
            "N_SUSY":     int(stats.n_susy),
            "N_RS_L":     int(stats.n_rs_L),
            "N_RS_R":     int(stats.n_rs_R),
            "N_RS":       int(stats.n_rs),
            "V_RR":       int(stats.n_v_rr),
            "H_RR":       int(stats.n_h_rr),
            "V_T":        int(stats.n_v_t),
            "H_T":        int(stats.n_h_t),
            "N_V":        int(stats.n_v),
            "N_H":        int(stats.n_h),
            "SPIN0":      int(stats.spin_counts_total.get("0",   0)),
            "SPIN_HALF":  int(stats.spin_counts_total.get("1/2", 0)),
            "SPIN1":      int(stats.spin_counts_total.get("1",   0)),
            "SPIN_3HALF": int(stats.spin_counts_total.get("3/2", 0)),
            "SPIN2":      int(stats.spin_counts_total.get("2",   0)),
            "INTERNAL_SYMMETRY":       internal_sym,
            "TWISTED_SUPERSECTORS":    _format_supersector_keys(stats.twisted_supersectors),
            "ASYM_TWIST_SUPERSECTORS": _format_supersector_keys(stats.rs_supersectors),
            "SYM_TWIST_SUPERSECTORS":  _format_supersector_keys(stats.vh_t_supersectors),
            "MULTIPLET_MATCH": mult_match_str,
            **{f"MULT_{k}": v for k, v in mult_counts.items()},
            "dt": time.time() - t0,
        }

    except Exception as e:
        return _blank_result(source, model_i, run_label, type_ii, susy_choice, struct_choice,
                             job_type, row_dict, f"Spectrum failed: {e}", time.time() - t0)


# ── Collapsed output ───────────────────────────────────────────────────────────

def _write_collapsed_csv(
    source: str,
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    out_dir: str,
    file_suffix: str = "SUSY",
    outcome_cols: Tuple[str, ...] = ("N_RS", "N_V", "N_H"),
) -> str:
    df_in_idx = df_in.reset_index(drop=True).copy()
    df_in_idx.insert(0, "model_i", df_in_idx.index + 1)
    # Cross with TYPE_II so IIA and IIB outcomes for the same model are kept as
    # separate rows rather than merged together (their outcomes are unrelated).
    df_in_idx = df_in_idx.merge(pd.DataFrame({"TYPE_II": ["IIA", "IIB"]}), how="cross")

    has_mult = "MULTIPLET_MATCH" in df_out.columns
    collapsed_rows: List[Dict[str, Any]] = []

    group_cols = ["model_i", "TYPE_II"] if "TYPE_II" in df_out.columns else ["model_i"]
    for key, grp in df_out.groupby(group_cols):
        key_t = key if isinstance(key, tuple) else (key,)
        row_key = dict(zip(group_cols, key_t))
        outcomes_df = (
            grp[list(outcome_cols)].drop_duplicates().sort_values(list(outcome_cols))
        )
        outcome_strs = [
            "[" + ",".join(str(int(row[c])) for c in outcome_cols) + "]"
            for _, row in outcomes_df.iterrows()
        ]
        mult_outcome_str = ""; n_distinct_mult = 0
        if has_mult:
            all_matches = grp["MULTIPLET_MATCH"].dropna().astype(str)
            meaningful  = sorted(s for s in all_matches if s and not s.startswith("0 ") and not s.startswith("no "))
            trivial     = sorted(s for s in all_matches if s.startswith("0 ") or s.startswith("no "))
            distinct    = list(dict.fromkeys(meaningful + trivial))
            n_distinct_mult  = len(distinct)
            mult_outcome_str = "; ".join(distinct)

        collapsed_rows.append({
            **{k: (int(v) if k == "model_i" else v) for k, v in row_key.items()},
            "N_SURVIVING_VARIANTS": len(grp),
            "N_DISTINCT_OUTCOMES": len(outcome_strs),
            "OUTCOMES_[N_RS,N_V,N_H]": "; ".join(outcome_strs),
            "N_DISTINCT_MULTIPLET_OUTCOMES": n_distinct_mult,
            "OUTCOMES_MULTIPLETS": mult_outcome_str,
        })

    empty_cols = group_cols + [
        "N_SURVIVING_VARIANTS", "N_DISTINCT_OUTCOMES",
        "OUTCOMES_[N_RS,N_V,N_H]", "N_DISTINCT_MULTIPLET_OUTCOMES", "OUTCOMES_MULTIPLETS",
    ]
    df_c = pd.DataFrame(collapsed_rows) if collapsed_rows else pd.DataFrame(columns=empty_cols)
    merge_cols = [c for c in group_cols if c in df_in_idx.columns]
    df_m = df_in_idx.merge(df_c, on=merge_cols, how="left")
    df_m["N_SURVIVING_VARIANTS"]          = df_m["N_SURVIVING_VARIANTS"].fillna(0).astype(int)
    df_m["N_DISTINCT_OUTCOMES"]           = df_m["N_DISTINCT_OUTCOMES"].fillna(0).astype(int)
    df_m["OUTCOMES_[N_RS,N_V,N_H]"]      = df_m["OUTCOMES_[N_RS,N_V,N_H]"].fillna("")
    df_m["N_DISTINCT_MULTIPLET_OUTCOMES"] = df_m["N_DISTINCT_MULTIPLET_OUTCOMES"].fillna(0).astype(int)
    df_m["OUTCOMES_MULTIPLETS"]           = df_m["OUTCOMES_MULTIPLETS"].fillna("")

    summary_cols = [
        "N_SURVIVING_VARIANTS", "N_DISTINCT_OUTCOMES", "OUTCOMES_[N_RS,N_V,N_H]",
        "N_DISTINCT_MULTIPLET_OUTCOMES", "OUTCOMES_MULTIPLETS",
    ]
    col_order = ["model_i", "TYPE_II"] + [c for c in df_in.columns if c in df_m.columns] + summary_cols
    df_m = df_m[[c for c in col_order if c in df_m.columns]]

    out_path = os.path.join(out_dir, f"{source}_spectra_stats_{file_suffix}_collapsed.csv")
    df_m.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    print(f"[ok] Collapsed: {out_path}  ({len(df_m)} rows)", flush=True)
    return out_path


# ── Driver ────────────────────────────────────────────────────────────────────

def main() -> None:
    dfs: Dict[str, pd.DataFrame] = {}
    for source, path in INPUT_FILES.items():
        try:
            dfs[source] = read_table_auto(path)
            print(f"[info] Loaded {source}: {len(dfs[source])} rows", flush=True)
        except FileNotFoundError:
            print(f"[warn] {path} not found — skipping {source}", flush=True)

    if not dfs:
        print("[error] No input files found.", flush=True)
        return

    jobs: List[Dict[str, Any]] = []

    for source, df in dfs.items():
        struct_variants = structural_variants(source)

        for r, row in df.iterrows():
            row_dict = {c: row.get(c, "") for c in df.columns}

            try:
                basis = BASIS_BUILDERS[source](pd.Series(row_dict))
            except Exception as e:
                print(f"[warn] {source} row {r+1}: basis build failed: {e}", flush=True)
                continue

            sb = susy_breaking_indices(basis)
            preserving = susy_preserving_choice(sb["left"], sb["right"])
            breaking   = susy_breaking_choices(sb["left"],  sb["right"])

            for type_ii in ("IIA", "IIB"):
                for struct in struct_variants:
                    # SUSY-preserving job
                    jobs.append({
                        "source": source, "model_i": r + 1, "type_ii": type_ii,
                        "row_dict": row_dict,
                        "susy_choice": preserving,
                        "struct_choice": struct,
                        "job_type": "SUSY",
                        "filter_susy": False,
                        "typeiiff_path": TYPEIIFF_PATH,
                        "mult_csv": MULT_CSV_PATH,
                        "write_processed": WRITE_PROCESSED,
                        "proc_dir": os.path.join(PROCESSED_DIR_SUSY, source, type_ii),
                        "max_n_susy": MAX_N_SUSY,
                    })
                    # SUSY-breaking jobs
                    if SCAN_NON_SUSY:
                        for bc in breaking:
                            jobs.append({
                                "source": source, "model_i": r + 1, "type_ii": type_ii,
                                "row_dict": row_dict,
                                "susy_choice": bc,
                                "struct_choice": struct,
                                "job_type": "non_SUSY",
                                "filter_susy": True,
                                "typeiiff_path": TYPEIIFF_PATH,
                                "mult_csv": MULT_CSV_PATH,
                                "write_processed": WRITE_PROCESSED,
                                "proc_dir": os.path.join(PROCESSED_DIR_NON_SUSY, source, type_ii),
                                "max_n_susy": MAX_N_SUSY,
                            })

    total         = len(jobs)
    n_susy_jobs   = sum(1 for j in jobs if j["job_type"] == "SUSY")
    n_non_susy    = total - n_susy_jobs

    if total == 0:
        print("[warn] No jobs scheduled.", flush=True)
        return

    print(f"\n[info] {total} total jobs ({n_susy_jobs} SUSY + {n_non_susy} non-SUSY) "
          f"across {MAX_WORKERS} workers", flush=True)
    for source, df in dfs.items():
        sv   = structural_variants(source)
        n_s  = len(sv)
        # count SUSY-breaking variants for first row (same for all rows of that class)
        row0 = {c: df.iloc[0].get(c, "") for c in df.columns}
        try:
            b0   = BASIS_BUILDERS[source](pd.Series(row0))
            sb0  = susy_breaking_indices(b0)
            n_b  = len(susy_breaking_choices(sb0["left"], sb0["right"])) if SCAN_NON_SUSY else 0
        except Exception:
            n_b  = 0
        non_susy_part = f" + {n_b} non-SUSY" if n_b else ""
        print(f"  {source:<15}: {len(df)} basis × 2 (IIA/IIB) × {n_s} struct × (1 SUSY-preserving{non_susy_part}) "
              f"= {len(df)*2*n_s*(1+n_b)} jobs", flush=True)

    results_susy:     Dict[str, List[Dict]] = {s: [] for s in dfs}
    results_non_susy: Dict[str, List[Dict]] = {s: [] for s in dfs}
    n_filtered = 0
    run_done   = 0
    t_all      = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_run_one_job, job) for job in jobs]
        for fut in as_completed(futures):
            run_done += 1
            res = fut.result()
            if res is None:
                n_filtered += 1
                continue
            src      = res["source"]
            jt       = res.get("job_type", "non_SUSY")
            if jt == "SUSY":
                results_susy[src].append(res)
            else:
                results_non_susy[src].append(res)

            if PRINT_EACH_RESULT:
                scalar = "H" if res.get("N_SUSY", 0) >= 2 else "S"
                print(f"\n[{run_done}/{total}] [{jt}] {res['run_label']}", flush=True)
                if res.get("err"):
                    print(f"  [ERR] {res['err']}", flush=True)
                mult = res.get("MULTIPLET_MATCH", "")
                print(
                    f"  {res.get('dt',0.):.2f}s  MI={res['MI_OK']}  "
                    f"N=(L={res['N_SUSY_L']},R={res['N_SUSY_R']},tot={res['N_SUSY']})  "
                    f"RS={res['N_RS']}  V={res['N_V']} {scalar}={res['N_H']}  "
                    f"spins=[{res['SPIN0']},{res['SPIN_HALF']},{res['SPIN1']},"
                    f"{res['SPIN_3HALF']},{res['SPIN2']}]"
                    + (f"  MULT={mult}" if mult else ""),
                    flush=True,
                )

    # ── Write output CSVs ──────────────────────────────────────────────────────
    for source, df_in in dfs.items():
        for job_type, out_rows in [("SUSY", results_susy[source]),
                                   ("non_SUSY", results_non_susy[source])]:
            if not out_rows:
                print(f"[info] {source} [{job_type}]: no results.", flush=True)
                continue

            df_out = pd.DataFrame(out_rows)

            base_cols = list(df_in.columns)
            for c in base_cols:
                if c not in df_out.columns:
                    df_out[c] = ""

            # Collect SUSY-breaking and structural phase columns present in results
            susy_cols = sorted(
                {c for c in df_out.columns if c.startswith("S_") or c.startswith("Sbar_")},
                key=lambda s: (0 if s.startswith("S_") else 1, int(s.split("_")[-1])),
            )
            s_cols = structural_phase_cols(source)
            for c in s_cols:
                if c not in df_out.columns:
                    df_out[c] = 0

            for c in _STATS_COLS:
                if c not in df_out.columns:
                    df_out[c] = "" if c in _STR_STATS else 0

            mult_cols = sorted(c for c in df_out.columns if c.startswith("MULT_"))
            for c in mult_cols:
                df_out[c] = df_out[c].fillna(0).astype(int)

            df_out["model_i"] = df_out["model_i"].astype(int)
            sort_cols = ["model_i", "TYPE_II"] + susy_cols + s_cols
            df_out    = df_out.sort_values(sort_cols, kind="mergesort")

            col_order = base_cols + susy_cols + s_cols + _STATS_COLS + mult_cols
            df_save   = df_out[[c for c in col_order if c in df_out.columns]].copy()

            out_path = os.path.join(OUT_DIR, f"{source}_spectra_stats_{job_type}.csv")
            df_save.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
            print(f"[ok] {source} [{job_type}]: {out_path}  ({len(df_save)} rows)", flush=True)

            try:
                summary = (
                    df_save.groupby(["N_SUSY_L", "N_SUSY_R"])
                    .size().reset_index(name="count")
                    .sort_values(["N_SUSY_L", "N_SUSY_R"])
                )
                print("  (N_L, N_R) breakdown:")
                for _, srow in summary.iterrows():
                    print(f"    (N_L={int(srow['N_SUSY_L'])}, N_R={int(srow['N_SUSY_R'])}): "
                          f"{int(srow['count'])} rows", flush=True)
            except Exception:
                pass

            _write_collapsed_csv(source, df_in, df_out, OUT_DIR, file_suffix=job_type)

    dt_all = time.time() - t_all
    print(f"\n[ok] Done in {dt_all:.2f}s  (filtered {n_filtered} non-SUSY runs "
          f"with N > {MAX_N_SUSY}).", flush=True)


if __name__ == "__main__":
    main()
