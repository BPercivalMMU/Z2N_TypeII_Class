# TypeIIFreeFermioniser.py  (v5)
# Updates from v3:
# - Distinguishes LEFT vs RIGHT supersymmetry in the Comments labels and
#   SpectrumStats fields ("gravitino_L"/"gravitino_R", "RS_L"/"RS_R") so that
#   SUSY-broken (N <= 2) models can be analysed correctly.  SpectrumStats now
#   exposes n_susy_L, n_susy_R, n_rs_L, n_rs_R (with n_susy and n_rs still
#   defined as the totals for backwards compatibility) and a spin_counts_total
#   dict for the full spin histogram (the natural primary output for N=0).
# - Supermultiplet matching is now available for N = 1..5.  At N=1 the library
#   contains the (4D) supergravity, Rarita-Schwinger, vector and chiral
#   multiplets:
#       G_N1     = [0, 0, 0, 1, 1]
#       RS_N1    = [0, 0, 1, 1, 0]
#       V_N1     = [0, 1, 1, 0, 0]
#       Chiral_N1= [2, 1, 0, 0, 0]
#   At N=0 no multiplet matching is attempted; the processed CSV reports
#   "N=0 — no supermultiplets; spin counts above are the full content."
# - write_processed_csv now records the (n_susy_L, n_susy_R, total) tuple
#   above the per-supersector spin breakdown.
#
# All v3 behaviour for the SUSY (N >= 2) case is preserved.  In particular:
#   Processed states are written in a COMPLEX-fermion representation.
#   Oscillators are collapsed to complex oscillators, e.g. psi1/psi2 -> psi12.
#   Ramond vacua are written as left/right products.
#   Internal complex pairs are taken consistently 
#   Rows that do not admit this complex-pair interpretation are dropped at the
#   processed stage.
#   Possible RS identification prompted by twisted sectors:
#       (0,8)  -> psi12 oscillator
#       (8,0)  -> psib12 oscillator
#   V_T/H_T flags identify key states in twisted N=2 (8,8) supersectors where  
#   scalar/vector are distinguished -> belonging to hyper/vector mult
#   Processed CSV records RS producing supersectors and V_T/H_T producing
#   twisted supersectors separately at the bottom.

from __future__ import annotations

import os
import csv
import time
import itertools
from dataclasses import dataclass
from math import comb as _comb
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from itertools import combinations


@dataclass(frozen=True)
class SpectrumStats:
    mi_ok: bool
    n_susy: int
    # SUSY-broken bookkeeping: separate L/R gravitino counts.
    # n_susy = n_susy_L + n_susy_R for backwards compatibility.
    n_susy_L: int
    n_susy_R: int
    n_rs: int
    # Side-resolved RS counts (RS from (8,0) sectors live on the right-Ramond side, etc.)
    n_rs_L: int
    n_rs_R: int

    n_v: int
    n_h: int

    n_v_rr: int
    n_h_rr: int
    n_v_t: int
    n_h_t: int

    # Spin counts for the WHOLE massless spectrum (sum across all supersectors).
    # These are computed from the "Spin" column of the processed DF and are the
    # main quantity to inspect for N=0 models, where supermultiplet matching is
    # not applicable.
    spin_counts_total: Dict[str, int]

    # diagnostics
    n_rs_sectors: int
    n_vh_t_sectors: int

    # supersector bookkeeping
    twisted_supersectors: List[Tuple[int, ...]]
    rs_supersectors: List[Tuple[int, ...]]
    vh_t_supersectors: List[Tuple[int, ...]]


# ── Supermultiplet matching helpers ─────────────────────────────────────────

def _sugra_multiplet_array(n_susy: int, h_max: float) -> List[int]:
    """Return spin-state count array [n0, n½, n1, n3/2, n2] for a massless
    N multiplet with highest helicity h_max.

    Convention
    ----------
    * Positive-helicity states are counted once.
    * Scalars (h=0) are doubled to account for the CPT-conjugate multiplet,
      EXCEPT when the multiplet is self-conjugate (h_max == n_susy / 4).

    Examples (confirmed)
    --------------------
    N=2, h_max=3/2  → [0, 1, 2, 1, 0]   (RS_N2)
    N=2, h_max=1    → [2, 2, 1, 0, 0]   (V_N2)
    N=4, h_max=2    → [2, 4, 6, 4, 1]   (G_N4)
    N=4, h_max=1    → [6, 4, 1, 0, 0]   (V_N4)
    """
    cpt_self = abs(h_max - n_susy / 4.0) < 1e-9
    spin_idx = {2.0: 4, 1.5: 3, 1.0: 2, 0.5: 1, 0.0: 0}
    arr = [0, 0, 0, 0, 0]
    for k in range(n_susy + 1):
        h = h_max - k / 2.0
        nk = _comb(n_susy, k)
        if h > 1e-9:
            idx = spin_idx.get(round(h * 2) / 2.0)
            if idx is not None:
                arr[idx] += nk
        elif abs(h) < 1e-9:
            arr[0] += nk if cpt_self else 2 * nk
        else:
            # h < 0: for non-self-conjugate multiplets, the negative-helicity state
            # has a CPT partner at |h| > 0 which must be counted as a positive-helicity
            # physical state.  
            if not cpt_self:
                h_abs = abs(h)
                idx = spin_idx.get(round(h_abs * 2) / 2.0)
                if idx is not None:
                    arr[idx] += nk
    return arr


def _multiplet_library(n_susy: int) -> List[Tuple[str, List[int]]]:
    """Return [(name, array)] for the standard massless multiplets at SUSY level n_susy.

    For N=1, the library is: SUGRA (supergravity), RS (Rarita-Schwinger), V (vector),
    and Chiral (= 2× h_max=½ multiplet, CPT self-conjugate = [2, 1, 0, 0, 0] × 2
    or rather, in our convention with scalars doubled for non-self-conjugate:
    the N=1 chiral multiplet has one Weyl fermion + one complex scalar = 2 real
    scalars and 1 fermion polarisation pair = [2, 1, 0, 0, 0]).
    For N=2, the library is: SUGRA, RS, V (vector), H (full hypermultiplet = [4,2,0,0,0]).
    For N=3,4,5, the library is: SUGRA, RS, V only — the hypermultiplet is not a standard
    short multiplet at these SUSY levels and is omitted to avoid spurious matches.

    The H_N2 entry is defined as 2 × the half-hypermultiplet (h_max=1/2, CPT self-
    conjugate), giving the physically relevant full hypermultiplet [4, 2, 0, 0, 0].
    """
    entries = []
    slots = [("SUGRA", 2.0), ("RS", 1.5), ("V", 1.0)]
    if n_susy == 2:
        slots.append(("H", 0.5))
    if n_susy == 1:
        # N=1 chiral multiplet: half-hyper-like, CPT self-conjugate at h_max=1/2
        # giving 1 Weyl fermion (=1 spin-1/2 state in positive-helicity counting)
        # + 1 complex scalar (=2 real scalars).
        slots.append(("Chiral", 0.5))
    for label, h_max in slots:
        arr = _sugra_multiplet_array(n_susy, h_max)
        if label == "H":
            # half-hyper → full hyper for N=2
            arr = [x * 2 for x in arr]
        if any(x > 0 for x in arr):
            entries.append((f"{label}_N{n_susy}", arr))
    return entries


def _match_multiplets_str(
    spin_counts: List[int],
    multiplets: List[Tuple[str, List[int]]],
) -> str:
    """Try to write spin_counts as a non-negative integer linear combination of
    the given multiplet arrays.  Returns a readable string such as
    ``'2*RS_N2 + 4*V_N2'`` or ``'no supermultiplet matching present'``.
    """
    if not multiplets or all(x == 0 for x in spin_counts):
        return "0 (empty sector)"

    names  = [m[0] for m in multiplets]
    arrays = [m[1] for m in multiplets]
    n_spin = len(spin_counts)

    # Upper bound on coefficient for each multiplet
    def max_c(arr: List[int]) -> int:
        bounds = [spin_counts[i] // arr[i] for i in range(n_spin) if arr[i] > 0]
        return min(bounds) if bounds else 0

    max_coeffs = [max_c(a) for a in arrays]

    # Depth-first search over coefficient combinations.
    # Iterate from max coefficient down to 0 (greedy) so that multiplets listed
    # earlier in the library are preferred.  With H_N2 before Half_H_N2 in the
    # CSV, this ensures full hypermultiplets are used first and half-hypers only
    # appear when no full-hyper solution exists.
    def dfs(idx: int, remaining: List[int]) -> Optional[List[int]]:
        if idx == len(arrays):
            return [] if all(r == 0 for r in remaining) else None
        arr = arrays[idx]
        mc = min(max_coeffs[idx],
                 min((remaining[i] // arr[i] for i in range(n_spin) if arr[i] > 0),
                     default=0))
        for c in range(mc, -1, -1):
            new_rem = [r - c * a for r, a in zip(remaining, arr)]
            result = dfs(idx + 1, new_rem)
            if result is not None:
                return [c] + result
        return None

    coeffs = dfs(0, list(spin_counts))
    if coeffs is None:
        return "no supermultiplet matching present"
    parts = [f"{c}*{n}" for c, n in zip(coeffs, names) if c > 0]
    return (" + ".join(parts)) if parts else "0 (all-zero)"


def load_multiplet_library_csv(
    csv_path: str, n_susy: int
) -> Optional[List[Tuple[str, List[int]]]]:
    """Load multiplet definitions for a given SUSY level from a CSV file.

    Expected columns: susy_level, name, n0, n_half, n1, n_3half, n2
    Any additional columns (e.g. notes) are ignored.
    Returns None if the file does not exist or has no rows for n_susy.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        df_n = df[df["susy_level"].astype(int) == int(n_susy)]
        if df_n.empty:
            return None
        result: List[Tuple[str, List[int]]] = []
        for _, row in df_n.iterrows():
            arr = [
                int(row["n0"]),
                int(row["n_half"]),
                int(row["n1"]),
                int(row["n_3half"]),
                int(row["n2"]),
            ]
            result.append((str(row["name"]), arr))
        return result
    except Exception:
        return None


class FreeFermionModel:
    def __init__(
        self,
        basis: np.ndarray,
        gso: np.ndarray,
        *,
        label: str = "",
        comp_dim: Optional[int] = None,
        type_ii: str = "IIB",
        multiplet_csv: Optional[str] = None,
    ):
        self.label = label or ""
        self.basis = np.array(basis, dtype=float)
        self.gso = np.array(gso, dtype=complex)
        self.nbv = self.basis.shape[0]

        self.type_ii = (type_ii or "IIB").strip().upper()
        if self.type_ii not in ("IIA", "IIB"):
            raise ValueError(f"type_ii must be 'IIA' or 'IIB'. Got {type_ii!r}")

        if comp_dim is None:
            self.comp_dim = (self.basis.shape[1] - 16) // 4
        else:
            self.comp_dim = int(comp_dim)

        if self.comp_dim < 0 or self.comp_dim > 8 or self.comp_dim % 2 != 0:
            raise ValueError(f"comp_dim must be even and between 0-8. Got {self.comp_dim}")

        self.LC_spacetime_dim = 8 - self.comp_dim
        self.multiplet_csv = multiplet_csv
        self._validate_inputs()

    @classmethod
    def from_files(
        cls,
        basis_file: str,
        gso_file: str,
        *,
        label: str = "",
        comp_dim: Optional[int] = None,
        type_ii: str = "IIB",
        multiplet_csv: Optional[str] = None,
    ) -> "FreeFermionModel":
        basis = np.loadtxt(basis_file)
        gso = np.loadtxt(gso_file)
        return cls(basis=basis, gso=gso, label=label, comp_dim=comp_dim, type_ii=type_ii,
                   multiplet_csv=multiplet_csv)

    @classmethod
    def from_arrays(
        cls,
        basis: np.ndarray,
        gso: np.ndarray,
        *,
        label: str = "",
        comp_dim: Optional[int] = None,
        type_ii: str = "IIB",
        multiplet_csv: Optional[str] = None,
    ) -> "FreeFermionModel":
        return cls(basis=basis, gso=gso, label=label, comp_dim=comp_dim, type_ii=type_ii,
                   multiplet_csv=multiplet_csv)

    # -------------------------- validation --------------------------

    def _validate_inputs(self) -> None:
        expected_cols = 16 + 4 * self.comp_dim
        if self.basis.ndim != 2:
            raise ValueError("Basis must be a 2D array.")
        if self.basis.shape[1] != expected_cols:
            raise ValueError(
                f"Basis matrix has incorrect shape. Expected {expected_cols} columns, got {self.basis.shape[1]}"
            )
        if self.nbv < 1:
            raise ValueError("Basis must contain at least b1.")
        if not np.all(self.basis[0] == 1):
            raise ValueError("Model inconsistency: first basis vector (b1) must be all 1s.")
        self.b1 = self.basis[0].copy()

        if self.gso.ndim != 2 or self.gso.shape[0] != self.gso.shape[1]:
            raise ValueError("GSO matrix must be square.")
        if self.gso.shape[0] != self.nbv:
            raise ValueError("GSO matrix size must match number of basis vectors.")

    # -------------------------- core math --------------------------

    def dot_prod(self, b1: np.ndarray, b2: np.ndarray) -> float:
        left_end = 8 + self.comp_dim * 2
        right_end = 16 + self.comp_dim * 4
        left = float(np.dot(b1[:left_end], b2[:left_end]))
        right = float(np.dot(b1[left_end:right_end], b2[left_end:right_end]))
        return left - right

    def calculate_sector_gso(
        self,
        sector: np.ndarray,
        b_sector: np.ndarray,
        sector_unred: np.ndarray,
        a: int,
        b: int,
    ) -> complex:
        left_end = 8 + self.comp_dim * 2

        s_delta1 = (-1) ** (sector[a][0] + sector[a][left_end])
        s_delta2 = (-1) ** (sector[b][0] + sector[b][left_end])

        sum_b = np.sum(b_sector[b]) - 1
        sum_a = np.sum(b_sector[a]) - 1
        s_gso1 = (s_delta1 ** sum_b) * (s_delta2 ** sum_a)

        diff = 0.5 * (sector_unred[a] - sector[a])
        s_gso2 = np.exp(-1j * np.pi * 0.5 * self.dot_prod(diff, sector_unred[b]))

        a_bsector = b_sector[a]
        b_bsector = b_sector[b]
        k_indices, l_indices = np.nonzero(np.outer(a_bsector, b_bsector))
        if len(k_indices) == 0:
            s_gso3 = 1.0 + 0j
        else:
            s_gso3 = np.prod(
                (self.gso[k_indices, l_indices] ** (a_bsector[k_indices] * b_bsector[l_indices]))
            )

        return complex(s_gso1) * complex(s_gso2) * complex(s_gso3)

    # -------------------------- modular invariance checks --------------------------

    def basis_prod_matrix(self) -> np.ndarray:
        BP = np.zeros((self.nbv, self.nbv))
        for i in range(self.nbv):
            for k in range(self.nbv):
                BP[i, k] = self.dot_prod(self.basis[i], self.basis[k])
        return BP

    def verify_basis_prod_matrix(self) -> bool:
        BP = self.basis_prod_matrix()
        ok = True
        for i in range(self.nbv):
            if BP[i, i] % 8 != 0:
                ok = False
        for i in range(self.nbv):
            for j in range(i + 1, self.nbv):
                if BP[i, j] % 4 != 0:
                    ok = False
        return ok

    def verify_basis_high_order(self) -> bool:
        ok = True
        for a, b, c in combinations(range(self.nbv), 3):
            inter3 = (self.basis[a] * self.basis[b]) * self.basis[c]
            n3 = self.dot_prod(inter3, self.b1)
            if n3 % 2 != 0:
                ok = False
        for a, b, c, d in combinations(range(self.nbv), 4):
            inter4 = ((self.basis[a] * self.basis[b]) * self.basis[c]) * self.basis[d]
            n4 = self.dot_prod(inter4, self.b1)
            if n4 % 2 != 0:
                ok = False
        return ok

    def verify_gso_invariance(self) -> bool:
        n1 = n2 = 0
        for i in range(self.nbv):
            for k in range(self.nbv):
                n_ik = self.dot_prod(self.basis[i], self.basis[k])
                if i == k:
                    lhs = np.round(self.gso[i, i])
                    rhs = np.round(np.exp(1j * np.pi * (n_ik / 8.0)) * np.conj(self.gso[i, 0]))
                    if lhs != rhs:
                        n1 += 1
                else:
                    lhs = np.round(self.gso[i, k])
                    rhs = np.round(np.exp(1j * np.pi * (n_ik / 4.0)) * np.conj(self.gso[k, i]))
                    if lhs != rhs:
                        n2 += 1
        return (n1 == 0 and n2 == 0)

    def _modular_invariance_diagnostics(self) -> List[str]:
        """Return a human-readable description of every MI violation found.

        Empty list → all checks pass and the model is consistent.

        Checks performed
        ----------------
        1. Basis self-products:   β_i·β_i ≡ 0 mod 8  (code) = 0 mod 4 (paper eq. 2.4,
           since code dot = 2×paper dot).
        2. Basis cross-products:  β_i·β_j ≡ 0 mod 4  (code) = 0 mod 2 (paper eq. 2.4).
        3. Three- and four-vector intersection conditions (paper eq. 2.4, last line).
        4. GGSO diagonal constraint:    C(β_i,β_i) = exp(iπ β_i²/8) C(β_i,β_1)*.
        5. GGSO off-diagonal constraint: C(β_i,β_j) C(β_j,β_i)* = exp(iπ β_i·β_j/4).
        """
        failures: List[str] = []
        BP = self.basis_prod_matrix()

        for i in range(self.nbv):
            v = int(round(BP[i, i]))
            if v % 8 != 0:
                failures.append(
                    f"Basis condition (eq. 2.4): β_{i}·β_{i} = {v} ≢ 0 mod 8 "
                    f"(paper: β·β = 0 mod 4; code dot = 2×paper dot)"
                )
        for i in range(self.nbv):
            for j in range(i + 1, self.nbv):
                v = int(round(BP[i, j]))
                if v % 4 != 0:
                    failures.append(
                        f"Basis condition (eq. 2.4): β_{i}·β_{j} = {v} ≢ 0 mod 4 "
                        f"(paper: β_a·β_b = 0 mod 2)"
                    )
        for a, b, c in combinations(range(self.nbv), 3):
            inter = (self.basis[a] * self.basis[b]) * self.basis[c]
            v = int(round(self.dot_prod(inter, self.b1)))
            if v % 2 != 0:
                failures.append(
                    f"Basis condition (eq. 2.4): |β_{a}∩β_{b}∩β_{c}| = {v} ≢ 0 mod 2"
                )
        for a, b, c, d in combinations(range(self.nbv), 4):
            inter = ((self.basis[a] * self.basis[b]) * self.basis[c]) * self.basis[d]
            v = int(round(self.dot_prod(inter, self.b1)))
            if v % 2 != 0:
                failures.append(
                    f"Basis condition (eq. 2.4): |β_{a}∩β_{b}∩β_{c}∩β_{d}| = {v} ≢ 0 mod 2"
                )
        for i in range(self.nbv):
            for k in range(self.nbv):
                n_ik = self.dot_prod(self.basis[i], self.basis[k])
                if i == k:
                    lhs = np.round(self.gso[i, i])
                    rhs = np.round(np.exp(1j * np.pi * (n_ik / 8.0)) * np.conj(self.gso[i, 0]))
                    if lhs != rhs:
                        failures.append(
                            f"GGSO condition (diagonal): C(β_{i},β_{i}) = {int(np.real(lhs))} "
                            f"but required = {int(np.real(rhs))} "
                            f"[C(β_a,β_a) = exp(iπ β_a²/8)·C(β_a,β_1)*]"
                        )
                else:
                    lhs = np.round(self.gso[i, k])
                    rhs = np.round(np.exp(1j * np.pi * (n_ik / 4.0)) * np.conj(self.gso[k, i]))
                    if lhs != rhs:
                        failures.append(
                            f"GGSO condition (off-diagonal): C(β_{i},β_{k}) = {int(np.real(lhs))} "
                            f"but required = {int(np.real(rhs))} "
                            f"[C(β_a,β_b)·C(β_b,β_a)* = exp(iπ β_a·β_b/4)]"
                        )
        return failures

    # -------------------------- sector generation / masses --------------------------

    def _calculate_sector_counts(self) -> Tuple[int, List[int]]:
        c_basis = np.zeros(self.nbv)
        for i in range(self.nbv):
            c_basis[i] = np.sum(self.basis[i] % 1 != 0)

        num_sec = 1
        rngs: List[int] = []
        for count in c_basis:
            if count == 0:
                num_sec *= 2
                rngs.append(2)
            else:
                num_sec *= 4
                rngs.append(4)
        return num_sec, rngs

    def generate_sectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_sec, rngs = self._calculate_sector_counts()

        sector = np.zeros((num_sec, self.basis.shape[1]))
        b_sector = np.zeros((num_sec, self.nbv))

        for idx, t in enumerate(itertools.product(*[range(r) for r in rngs])):
            sector[idx] = sum([self.basis[j] * t[j] for j in range(len(t))])
            b_sector[idx] = t

        sector_unred = sector.copy()
        sector_unred[0] = 2
        sector = sector % 2
        sector[sector == 1.5] = -0.5

        b_sector[0, 0] = 2
        return sector, b_sector.astype(int), sector_unred

    def calculate_sector_masses(self, sector: np.ndarray) -> np.ndarray:
        masses = np.zeros((sector.shape[0], 2))
        sl = sector.copy()
        sr = sector.copy()

        left_end = 8 + self.comp_dim * 2
        sl[:, left_end:] = 0
        sr[:, :left_end] = 0

        for i in range(sector.shape[0]):
            masses[i, 0] = -0.5 + self.dot_prod(sl[i], sl[i]) / 16.0
            masses[i, 1] = -0.5 - self.dot_prod(sr[i], sr[i]) / 16.0
        return masses

    # -------------------------- labels --------------------------

    def fermion_labels(self) -> List[str]:
        d = self.comp_dim
        Dm2 = 8 - d

        labels: List[str] = []
        for i in range(1, Dm2 + 1):
            labels.append(f"psi{i}")
        for i in range(1, d + 1):
            labels.append(f"chi{i}")
        for i in range(1, d + 1):
            labels.append(f"y{i}")
        for i in range(1, d + 1):
            labels.append(f"w{i}")

        for i in range(1, Dm2 + 1):
            labels.append(f"psib{i}")
        for i in range(1, d + 1):
            labels.append(f"chib{i}")
        for i in range(1, d + 1):
            labels.append(f"yb{i}")
        for i in range(1, d + 1):
            labels.append(f"wb{i}")

        return labels

    def compute_internal_symmetry_groups(self) -> dict:
        """
        Groups the 24 internal real fermions (y/w on the left; yb/wb on the right)
        by their boundary-condition signature across all basis vectors.  Fermions that
        share the same BC pattern are always simultaneously periodic/aperiodic in every
        sector, so they can form genuine complex-fermion pairs.

        Returns a dict with:
          'sym_str'       – readable symmetry string
          'groups'        – list of group dicts for display
          'll_pairs'      – [(name_a, idx_a, name_b, idx_b, cname)]  genuine left-left pairs
          'rr_pairs'      – [(name_a, idx_a, name_b, idx_b, cname)]  genuine right-right pairs
          'lr_pairs'      – [(name_L, idx_L, name_R, idx_R, cname)]  left-right (Ising) pairs
          'left_singles'  – [(name, idx)]  unpaired left fermions
          'right_singles' – [(name, idx)]  unpaired right fermions
        """
        import re
        from collections import defaultdict

        d    = self.comp_dim
        Dm2  = 8 - d
        left_end = 8 + 2 * d

        # --- internal fermion names and array indices ---
        left_internal: List[Tuple[str, int]] = []
        for i in range(1, d + 1):
            left_internal.append((f"y{i}",  Dm2 + d       + (i - 1)))
        for i in range(1, d + 1):
            left_internal.append((f"w{i}",  Dm2 + 2 * d   + (i - 1)))

        right_internal: List[Tuple[str, int]] = []
        for i in range(1, d + 1):
            right_internal.append((f"yb{i}", left_end + Dm2 + d       + (i - 1)))
        for i in range(1, d + 1):
            right_internal.append((f"wb{i}", left_end + Dm2 + 2 * d   + (i - 1)))

        def bc_sig(idx: int) -> tuple:
            return tuple(float(self.basis[k, idx]) for k in range(self.nbv))

        sig_to_left:  dict = defaultdict(list)
        sig_to_right: dict = defaultdict(list)
        for name, idx in left_internal:
            sig_to_left[bc_sig(idx)].append((name, idx))
        for name, idx in right_internal:
            sig_to_right[bc_sig(idx)].append((name, idx))

        all_sigs = sorted(set(sig_to_left.keys()) | set(sig_to_right.keys()))

        def make_cname(na: str, nb: str) -> str:
            ma = re.match(r"([a-z]+)(\d+)", na)
            mb = re.match(r"([a-z]+)(\d+)", nb)
            if ma and mb and ma.group(1) == mb.group(1):
                return ma.group(1) + ma.group(2) + mb.group(2)
            return na + nb

        ll_pairs:     List[tuple] = []
        rr_pairs:     List[tuple] = []
        left_singles_raw:  List[Tuple[str, int]] = []
        right_singles_raw: List[Tuple[str, int]] = []
        groups: List[dict] = []

        for sig in all_sigs:
            lf = sig_to_left.get(sig, [])
            rf = sig_to_right.get(sig, [])
            kL, kR = len(lf), len(rf)
            so_l = f"SO({kL})" if kL >= 2 else ("O(1)" if kL == 1 else None)
            so_r = f"SO({kR})" if kR >= 2 else ("O(1)" if kR == 1 else None)

            for i in range(0, len(lf) - 1, 2):
                na, ia = lf[i]; nb, ib = lf[i + 1]
                ll_pairs.append((na, ia, nb, ib, make_cname(na, nb)))
            if len(lf) % 2 == 1:
                left_singles_raw.append(lf[-1])

            for i in range(0, len(rf) - 1, 2):
                na, ia = rf[i]; nb, ib = rf[i + 1]
                rr_pairs.append((na, ia, nb, ib, make_cname(na, nb)))
            if len(rf) % 2 == 1:
                right_singles_raw.append(rf[-1])

            groups.append({
                "signature": sig, "left_fermions": lf, "right_fermions": rf,
                "so_left": so_l, "so_right": so_r,
            })

        # Try to form LR pairs from leftover singles (same BC signature)
        lr_pairs:      List[tuple] = []
        left_singles:  List[Tuple[str, int]] = []
        right_singles: List[Tuple[str, int]] = []
        sig_left_sing  = {bc_sig(idx): (n, idx) for n, idx in left_singles_raw}
        sig_right_sing = {bc_sig(idx): (n, idx) for n, idx in right_singles_raw}
        for sig in all_sigs:
            ls = sig_left_sing.get(sig)
            rs = sig_right_sing.get(sig)
            if ls and rs:
                na, ia = ls; nb, ib = rs
                lr_pairs.append((na, ia, nb, ib, make_cname(na, nb)))
            elif ls:
                left_singles.append(ls)
            elif rs:
                right_singles.append(rs)

        # Build symmetry string
        lf_strs = sorted([g["so_left"]  for g in groups if g["so_left"]],
                         key=lambda s: -(int(s[3:-1]) if s.startswith("SO") else 1))
        rf_strs = sorted([g["so_right"] for g in groups if g["so_right"]],
                         key=lambda s: -(int(s[3:-1]) if s.startswith("SO") else 1))
        l_part = " x ".join(lf_strs) if lf_strs else "1"
        r_part = " x ".join(rf_strs) if rf_strs else "1"
        sym_str = f"[{l_part}]_L  x  [{r_part}]_R"

        return {
            "sym_str":        sym_str,
            "groups":         groups,
            "ll_pairs":       ll_pairs,
            "rr_pairs":       rr_pairs,
            "lr_pairs":       lr_pairs,
            "left_singles":   left_singles,
            "right_singles":  right_singles,
        }

    # -------------------------- massless spectrum --------------------------

    @staticmethod
    def _phase(x: float | np.ndarray) -> complex | np.ndarray:
        return np.exp(1j * np.pi * x)

    def massless_raw(self) -> pd.DataFrame:
        sectors, b_sectors, sectors_unred = self.generate_sectors()
        masses = self.calculate_sector_masses(sectors)
        labels = self.fermion_labels()

        sector_cases = [
            {"indices": np.where((masses[:, 0] == -0.5) & (masses[:, 1] == -0.5))[0], "type": (0, 0)},
            {"indices": np.where((masses[:, 0] == 0.0)  & (masses[:, 1] == -0.5))[0], "type": (8, 0)},
            {"indices": np.where((masses[:, 0] == -0.5) & (masses[:, 1] == 0.0))[0],  "type": (0, 8)},
            {"indices": np.where((masses[:, 0] == 0.0)  & (masses[:, 1] == 0.0))[0],  "type": (8, 8)},
        ]

        bvec_indices = np.where(np.sum(b_sectors, axis=1) == 1)[0]

        left_end = 8 + 2 * self.comp_dim
        right_end = 16 + 4 * self.comp_dim

        left_pairs = [(i, i + 1) for i in range(0, 8, 2)]
        right_start = 8 + 2 * self.comp_dim
        right_pairs = [(i, i + 1) for i in range(right_start, right_start + 8, 2)]

        chi_pairs: List[Tuple[int, int]] = []
        if self.comp_dim >= 2:
            Dm2 = 8 - self.comp_dim
            chi_base_L = Dm2
            chi_base_R = (8 + 2 * self.comp_dim) + Dm2
            for p in range(1, self.comp_dim // 2 + 1):
                i1 = 2 * p - 1
                i2 = 2 * p
                chi_pairs.append((chi_base_L + (i1 - 1), chi_base_L + (i2 - 1)))
                chi_pairs.append((chi_base_R + (i1 - 1), chi_base_R + (i2 - 1)))

        # Genuine group-based y/w pairs: fermions in the same BC group always
        # share the same periodicity, so they form proper complex fermions (sa == sb).
        sym_info = self.compute_internal_symmetry_groups()
        yw_pairs_idx: List[Tuple[int, int]] = (
            [(ia, ib) for _, ia, _, ib, _ in sym_info["ll_pairs"]] +
            [(ia, ib) for _, ia, _, ib, _ in sym_info["rr_pairs"]]
        )
        all_pairs = left_pairs + right_pairs + chi_pairs + yw_pairs_idx

        results: List[Dict[str, Any]] = []

        for case in sector_cases:
            for sec_idx in case["indices"]:
                current_sec = sectors[sec_idx]
                delta_sec = ((-1) ** current_sec[0]) * ((-1) ** current_sec[left_end])

                variable_indices = np.where(current_sec == 1)[0]
                spin_vac_template = np.zeros_like(current_sec)

                possible_spin_vacs: List[np.ndarray] = []
                for rvals in itertools.product([0.0, -0.5], repeat=len(variable_indices)):
                    spin_vac = spin_vac_template.copy()
                    spin_vac[variable_indices] = rvals
                    if all(spin_vac[a] == spin_vac[b] for (a, b) in all_pairs):
                        possible_spin_vacs.append(spin_vac)

                def survives_all_projections(cand: np.ndarray) -> bool:
                    for v in bvec_indices:
                        exponent = np.sum(sectors[v] * cand)
                        lhs = self._phase(exponent)
                        rhs = complex(delta_sec) * self.calculate_sector_gso(
                            sectors, b_sectors, sectors_unred, sec_idx, v
                        )
                        if not np.isclose(lhs, rhs):
                            return False
                    return True

                valid_states: List[np.ndarray] = []

                if case["type"] == (0, 0):
                    for spin_vac in possible_spin_vacs:
                        for lo in range(0, left_end):
                            for ro in range(left_end, right_end):
                                cand = spin_vac.copy()
                                cand[lo] = 1.0
                                cand[ro] = 1.0
                                if survives_all_projections(cand):
                                    valid_states.append(cand)

                elif case["type"] == (8, 0):
                    for spin_vac in possible_spin_vacs:
                        for ro in range(left_end, right_end):
                            cand = spin_vac.copy()
                            cand[ro] = 1.0
                            if survives_all_projections(cand):
                                valid_states.append(cand)

                elif case["type"] == (0, 8):
                    for spin_vac in possible_spin_vacs:
                        for lo in range(0, left_end):
                            cand = spin_vac.copy()
                            cand[lo] = 1.0
                            if survives_all_projections(cand):
                                valid_states.append(cand)

                elif case["type"] == (8, 8):
                    for spin_vac in possible_spin_vacs:
                        cand = spin_vac.copy()
                        if survives_all_projections(cand):
                            valid_states.append(cand)

                for state in valid_states:
                    row: Dict[str, Any] = {
                        "Sector": np.array2string(b_sectors[sec_idx], suppress_small=False),
                        "(α_L,α_R)": case["type"],
                    }
                    for j, lab in enumerate(labels):
                        if state[j] != 0 or current_sec[j] == 1:
                            row[lab] = float(state[j])
                        else:
                            row[lab] = ""
                    results.append(row)

        return pd.DataFrame(results)

    # -------------------------- processing / grouping / stats --------------------------

    def processed_from_raw(
        self, df_raw: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[Tuple[int, ...]], List[Tuple[int, ...]], List[Tuple[int, ...]]]:
        out_cols = [
            "Sector",
            "State",
            "(α_L,α_R)",
            "Comments",
            "Spin",
            "super_key",
            "is_rs_sector",
            "is_vh_t_sector",
        ]
        if df_raw.empty:
            empty = pd.DataFrame(columns=out_cols)
            return empty, [], [], []

        cols = list(df_raw.columns)
        state_cols = [c for c in cols if c not in ("Sector", "(α_L,α_R)")]

        df = df_raw.copy()
        df[state_cols] = df[state_cols].apply(pd.to_numeric, errors="coerce")

        def parse_sector(x: Any) -> Tuple[int, ...]:
            if isinstance(x, str):
                vals = x.strip("[]").split()
                return tuple(int(v) for v in vals)
            if hasattr(x, "__iter__"):
                return tuple(int(v) for v in x)
            raise ValueError(f"Unrecognized Sector format: {x}")

        df["sector_tuple"] = df["Sector"].apply(parse_sector)

        vec_len = len(df["sector_tuple"].iloc[0])

        def supersector_key(t: Tuple[int, ...]) -> Tuple[int, ...]:
            if len(t) >= 3:
                return (t[0],) + t[3:]
            return t

        df["super_key"] = df["sector_tuple"].apply(supersector_key)

        untwisted_bases = [
            (2,) + (0,) * (vec_len - 1),
            (0, 1, 0) + (0,) * (vec_len - 3),
            (0, 0, 1) + (0,) * (vec_len - 3),
            (0, 1, 1) + (0,) * (vec_len - 3),
        ]
        if vec_len >= 3:
            untwisted_superkeys = {(t[0],) + t[3:] for t in untwisted_bases}
        else:
            untwisted_superkeys = set(untwisted_bases)

        labels = self.fermion_labels()
        idx_map = {lab: j for j, lab in enumerate(labels)}
        left_end = 8 + 2 * self.comp_dim
        Dm2 = 8 - self.comp_dim

        # ---------------- helpers ----------------

        def sector_vec_from_bsector(st: Tuple[int, ...]) -> np.ndarray:
            sec = np.zeros(self.basis.shape[1], dtype=float)
            for i, ti in enumerate(st):
                sec += self.basis[i] * float(ti)
            sec = sec % 2
            sec[sec == 1.5] = -0.5
            return sec

        def a_from_bc(x: float) -> int:
            if np.isclose(x, -0.5):
                return 1
            return int(round(float(x)))

        def sector_delta(sec_vec: np.ndarray) -> int:
            a_psi1 = a_from_bc(sec_vec[0])
            a_psib1 = a_from_bc(sec_vec[left_end])
            return int(((-1) ** (a_psi1 + a_psib1)))

        def is_plus_helicity(val: Any) -> bool:
            return val == 0 or (isinstance(val, (float, int)) and np.isclose(val, 0.0))

        def hel_to_sign(val: Any) -> Optional[int]:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            try:
                fv = float(val)
            except Exception:
                return None
            if np.isclose(fv, 0.0):
                return +1
            if np.isclose(fv, -0.5):
                return -1
            return None

        def sign_char_from_val(val: Any) -> Optional[str]:
            s = hel_to_sign(val)
            if s is None:
                return None
            return "+" if s == +1 else "-"

        def is_rr_sector(st: Tuple[int, ...]) -> bool:
            if len(st) < 3:
                return False
            if not (st[0] == 0 and st[1] == 1 and st[2] == 1):
                return False
            return all(v == 0 for v in st[3:])

        def rr_gate(row: pd.Series, sec_vec: np.ndarray) -> bool:
            if a_from_bc(sec_vec[0]) != 1:
                return False
            if a_from_bc(sec_vec[left_end]) != 1:
                return False
            if hel_to_sign(row.get("psi1")) is None:
                return False
            if hel_to_sign(row.get("psib1")) is None:
                return False
            return True

        def chi_pair_periodic(sec_vec: np.ndarray, side: str, k: int) -> bool:
            i1 = 2 * k - 1
            i2 = 2 * k
            a = f"chi{i1}" if side == "L" else f"chib{i1}"
            b = f"chi{i2}" if side == "L" else f"chib{i2}"
            ia = idx_map.get(a)
            ib = idx_map.get(b)
            if ia is None or ib is None:
                return False
            return (a_from_bc(sec_vec[ia]) == 1) and (a_from_bc(sec_vec[ib]) == 1)

        def chi_k_sign(row: pd.Series, side: str, k: int) -> Optional[int]:
            i1 = 2 * k - 1
            i2 = 2 * k
            a = f"chi{i1}" if side == "L" else f"chib{i1}"
            b = f"chi{i2}" if side == "L" else f"chib{i2}"
            sa = hel_to_sign(row.get(a))
            sb = hel_to_sign(row.get(b))
            if sa is None or sb is None:
                return None
            return int(sa * sb)

        def sigma_from_psi_chi(row: pd.Series, sec_vec: np.ndarray) -> int:
            spsi = hel_to_sign(row.get("psi1"))
            spsib = hel_to_sign(row.get("psib1"))
            if spsi is None or spsib is None:
                return +1

            sigma = int(spsi * spsib)
            for k in (1, 2, 3):
                if chi_pair_periodic(sec_vec, "L", k) and chi_pair_periodic(sec_vec, "R", k):
                    sL = chi_k_sign(row, "L", k)
                    sR = chi_k_sign(row, "R", k)
                    if sL is None or sR is None:
                        continue
                    sigma *= int(sL * sR)
            return int(sigma)

        # Compute group-based genuine complex-fermion pairs for internal fermions.
        # Fermions in the same BC group are always simultaneously periodic/aperiodic,
        # so sa == sb is physically enforced for those pairs.
        sym_info = self.compute_internal_symmetry_groups()
        ll_pairs_info = sym_info["ll_pairs"]   # (na, ia, nb, ib, cname)
        rr_pairs_info = sym_info["rr_pairs"]
        lr_pairs_info = sym_info["lr_pairs"]
        left_singles_info  = sym_info["left_singles"]   # (name, idx)
        right_singles_info = sym_info["right_singles"]

        # complex fermion pairs for processed representation
        def pair_defs_left() -> List[Tuple[str, Tuple[str, str]]]:
            defs: List[Tuple[str, Tuple[str, str]]] = []
            for i in range(1, Dm2, 2):
                defs.append((f"psi{i}{i+1}", (f"psi{i}", f"psi{i+1}")))
            for i in range(1, self.comp_dim, 2):
                defs.append((f"chi{i}{i+1}", (f"chi{i}", f"chi{i+1}")))
            # Group-based genuine LL pairs (replaces hardcoded y12,y34,… w12,…)
            for na, _, nb, _, cname in ll_pairs_info:
                defs.append((cname, (na, nb)))
            return defs

        def pair_defs_right() -> List[Tuple[str, Tuple[str, str]]]:
            defs: List[Tuple[str, Tuple[str, str]]] = []
            for i in range(1, Dm2, 2):
                defs.append((f"psib{i}{i+1}", (f"psib{i}", f"psib{i+1}")))
            for i in range(1, self.comp_dim, 2):
                defs.append((f"chib{i}{i+1}", (f"chib{i}", f"chib{i+1}")))
            # Group-based genuine RR pairs
            for na, _, nb, _, cname in rr_pairs_info:
                defs.append((cname, (na, nb)))
            return defs

        pair_defs_L = pair_defs_left()
        pair_defs_R = pair_defs_right()
        # LR pairs also contribute oscillator labels
        lr_pair_defs = [(cname, (na, nb)) for na, _, nb, _, cname in lr_pairs_info]
        all_pair_defs = pair_defs_L + pair_defs_R + lr_pair_defs

        def complex_oscillators(row: pd.Series) -> List[str]:
            out: List[str] = []
            for cname, (a, b) in all_pair_defs:
                va = row.get(a)
                vb = row.get(b)
                if cname.startswith("psi") or cname.startswith("chi"):
                    # Spacetime psi/psib pairs and internal chi/chib pairs:
                    # first member = fundamental (cname), second member = conjugate (cname+"c").
                    # For psi: distinguishes h=+2 graviton from dilaton/B-field scalars and
                    # suppresses the h=-2 CPT graviton.
                    # For chi: distinguishes complex-structure modulus (chi12⊗chib12) from
                    # Kähler modulus (chi12⊗chib12c); chi12c states are CPT partners and
                    # will be suppressed in compute_spin.
                    if va == 1:
                        out.append(cname)
                    elif vb == 1:
                        out.append(cname + "c")
                else:
                    # Internal (y, w, LR) pairs: both members map to the same label.
                    if va == 1 or vb == 1:
                        out.append(cname)
            # Unpaired singles
            for name, _ in left_singles_info + right_singles_info:
                if row.get(name) == 1:
                    out.append(name)
            return out

        def complex_vacuum_signs(row: pd.Series, side: str) -> Optional[List[str]]:
            # All entries in pair_defs_L / pair_defs_R are genuine complex pairs:
            # psi/chi (standard) and group-based LL/RR.  All require sa == sb.
            defs = pair_defs_L if side == "L" else pair_defs_R
            out: List[str] = []
            for _, (a, b) in defs:
                sa = sign_char_from_val(row.get(a))
                sb = sign_char_from_val(row.get(b))
                if sa is None and sb is None:
                    continue
                if sa is None or sb is None or sa != sb:
                    return None      # genuine complex pair: must be consistent
                out.append(sa)

            # LR (Ising) pairs are excluded from the vacuum ket display:
            # they are individual fermions with different BC signatures that
            # happen to be Ramond in some sectors, but they do not form genuine
            # complex pairs with anything on the same side.  Their helicities do
            # not enter the sigma/V_T/H_T calculation, so omitting them keeps the
            # ket compact and readable (4 entries = psi12, chi12, and any LL pairs).

            # Unpaired singles contribute individually
            singles = left_singles_info if side == "L" else right_singles_info
            for name, _ in singles:
                s = sign_char_from_val(row.get(name))
                if s is not None:
                    out.append(s)

            return out

        def lr_vacuum_signs(row: pd.Series) -> Optional[List[str]]:
            """Return one sign per Ramond LR (Ising) pair: '+' if both components
            share the same helicity, '-' if they are opposite.  Returns an empty
            list when no LR pair is Ramond.  Returns None on inconsistency (one
            side Ramond, the other not — should not happen with BC-matched pairs)."""
            out: List[str] = []
            for na, _, nb, _, _ in lr_pairs_info:
                sa = hel_to_sign(row.get(na))
                sb = hel_to_sign(row.get(nb))
                if sa is None and sb is None:
                    continue                 # pair is NS in this sector — skip
                if sa is None or sb is None:
                    return None             # unexpected inconsistency
                out.append("+" if sa * sb > 0 else "-")
            return out

        def collapse_row_complex(row: pd.Series) -> Optional[str]:
            osc  = complex_oscillators(row)
            vacL = complex_vacuum_signs(row, "L")
            vacR = complex_vacuum_signs(row, "R")
            vacLR = lr_vacuum_signs(row)

            if vacL is None or vacR is None or vacLR is None:
                return None

            parts: List[str] = []
            if osc:
                parts.append(" ".join(osc))

            vac_parts: List[str] = []
            if vacL:
                vac_parts.append("| " + " ".join(vacL) + ">_L")
            if vacLR:
                vac_parts.append("| " + " ".join(vacLR) + ">")   # LR Ising ket, no side suffix
            if vacR:
                vac_parts.append("| " + " ".join(vacR) + ">_R")

            if vac_parts:
                parts.append(" x ".join(vac_parts))

            state_str = " ".join(parts).strip()
            return state_str if state_str else None

        df["State"] = df.apply(collapse_row_complex, axis=1)
        df = df[df["State"].notna()].copy()
        df = df.drop_duplicates(subset=["Sector", "State", "(α_L,α_R)"]).copy()

        def oscillator_set(row: pd.Series) -> set[str]:
            return set(complex_oscillators(row))

        def is_S_sector(st: Tuple[int, ...]) -> bool:
            return len(st) >= 3 and st[0] == 0 and st[1] == 1 and st[2] == 0 and all(v == 0 for v in st[3:])

        def is_Sbar_sector(st: Tuple[int, ...]) -> bool:
            return len(st) >= 3 and st[0] == 0 and st[1] == 0 and st[2] == 1 and all(v == 0 for v in st[3:])

        def is_rs_sector_row(row: pd.Series) -> bool:
            st: Tuple[int, ...] = row["sector_tuple"]
            alpha_t = row["(α_L,α_R)"]

            if alpha_t not in ((8, 0), (0, 8)):
                return False
            if is_S_sector(st) or is_Sbar_sector(st):
                return False

            sec_vec = sector_vec_from_bsector(st)
            delt = sector_delta(sec_vec)
            a_psi1 = a_from_bc(sec_vec[0])
            a_psib1 = a_from_bc(sec_vec[left_end])

            if alpha_t == (8, 0):
                return (delt == -1) and (a_psi1 == 1)
            if alpha_t == (0, 8):
                return (delt == -1) and (a_psib1 == 1)
            return False

        df["is_rs_sector"] = df.apply(is_rs_sector_row, axis=1)

        def comment_row_pre(row: pd.Series) -> str:
            st: Tuple[int, ...] = row["sector_tuple"]
            alpha_t = row.get("(α_L,α_R)")
            sec_vec = sector_vec_from_bsector(st)
            osc = oscillator_set(row)

            psi1 = row.get("psi1")
            psib1 = row.get("psib1")

            if alpha_t == (0, 0):
                if osc == {"psi12", "psib12"}:
                    return "graviton"

            if is_S_sector(st):
                if is_plus_helicity(psi1):
                    if "psib12" in osc:
                        # Gravitino from the S-sector lives on the "left" SUSY side
                        # (its supercharge is generated by S). Tag separately so we
                        # can compute n_susy_L vs n_susy_R for SUSY-broken models.
                        return "gravitino_L"
            if is_Sbar_sector(st):
                if is_plus_helicity(psib1):
                    if "psi12" in osc:
                        return "gravitino_R"

            if alpha_t in ((8, 0), (0, 8)):
                delt = sector_delta(sec_vec)
                a_psi1 = a_from_bc(sec_vec[0])
                a_psib1 = a_from_bc(sec_vec[left_end])

                if alpha_t == (8, 0):
                    if (delt == -1) and (a_psi1 == 1):
                        if is_plus_helicity(psi1) and ("psib12" in osc):
                            # Spin-3/2 state from a (8,0) sector (= left spacetime Ramond),
                            # so the spacetime Rarita-Schwinger fermion is left-handed.
                            # In SUSY-preserving models this is the "left RS multiplet";
                            # in SUSY-broken models it can still survive even when left
                            # SUSY is broken (see paper §6.6, Table 18).
                            return "RS_L"

                elif alpha_t == (0, 8):
                    if (delt == -1) and (a_psib1 == 1):
                        if is_plus_helicity(psib1) and ("psi12" in osc):
                            # Spin-3/2 from (0,8) sector — right-handed RS fermion.
                            # As above, can survive even with right SUSY broken.
                            return "RS_R"

            if alpha_t == (8, 8):
                if rr_gate(row, sec_vec):
                    if is_plus_helicity(psi1):
                        sigma = sigma_from_psi_chi(row, sec_vec)
                        is_vector = (sigma == +1) if self.type_ii == "IIB" else (sigma == -1)

                        if is_rr_sector(st):
                            return "V_RR" if is_vector else "H_RR"
                        else:
                            return "V_T" if is_vector else "H_T"

            return ""

        df["Comments"] = df.apply(comment_row_pre, axis=1)

        # Compute Spin BEFORE comment suppression so V_T/H_T states retain their spin
        def compute_spin(row: pd.Series) -> str:
            comment = str(row.get("Comments", "")).strip()
            state = str(row.get("State", ""))

            # Use Comments only for unambiguous spin-2 and spin-3/2 identifications.
            # V_RR/V_T and H_RR/H_T are no longer short-circuited here: spin for (8,8)
            # sector states is derived directly from the sigma computation below so that
            # it is independent of the Comments column.
            if comment == "graviton":
                return "2"
            if comment in (
                "gravitino", "gravitino_L", "gravitino_R",
                "RS", "RS_L", "RS_R",
            ):
                return "3/2"

            # Infer spin from the actual sector vector.
            # α=(8,0) or (0,8) only means 8 periodic fermions on that side, but those
            # fermions may be internal (chi/y/w) rather than the spacetime psi/psib.
            # Only psi1/psi2 (indices 0,1) and psib1/psib2 (indices left_end, left_end+1)
            # contribute to spacetime spin.
            st: Tuple[int, ...] = row["sector_tuple"]
            sec_vec = sector_vec_from_bsector(st)
            left_st_ramond  = (a_from_bc(sec_vec[0]) == 1) or (a_from_bc(sec_vec[1]) == 1)
            right_st_ramond = (a_from_bc(sec_vec[left_end]) == 1) or (a_from_bc(sec_vec[left_end + 1]) == 1)

            # Oscillators appear before any "|" in the State string
            osc_part = state.split("|")[0].strip() if "|" in state else state
            osc_tokens = set(osc_part.split())
            has_psi12   = "psi12"   in osc_tokens
            has_psi12c  = "psi12c"  in osc_tokens
            has_psib12  = "psib12"  in osc_tokens
            has_psib12c = "psib12c" in osc_tokens

            if not left_st_ramond and not right_st_ramond:
                # Both sides NS in spacetime.
                # psi12  ⊗ psib12:  h = +1+1 = +2  → graviton
                # psi12c ⊗ psib12c: h = -1-1 = -2  → CPT graviton, suppress
                # psi12  ⊗ psib12c or psi12c ⊗ psib12: h = 0  → dilaton/B-field scalar
                # single psi12 or psib12 (+ chi/yw on the other side): h = ±1 → vector
                # chi/yw ⊗ chi/yw: h = 0 → scalar
                has_any_psi  = has_psi12 or has_psi12c
                has_any_psib = has_psib12 or has_psib12c
                if has_psi12 and has_psib12:
                    return "2"
                elif has_psi12c and has_psib12c:
                    return ""    # h=-2: CPT graviton, suppress
                elif has_any_psi and has_any_psib:
                    return "0"   # h=0: dilaton / B-field
                elif has_psi12 or has_psib12:
                    return "1"   # h=+1 single spacetime oscillator
                elif has_psi12c or has_psib12c:
                    return ""    # h=-1: suppress
                else:
                    return "0"   # chi/yw oscillators (including chi12c): scalar
            elif left_st_ramond and not right_st_ramond:
                # Left spacetime Ramond (h_L = ±½), right NS oscillator.
                # Suppress psi1='-' states (h_L=-½ gives h_{4D} ≤ +½; only psib12 would
                # give h=+½ but that is the "wrong" partner not counted in the multiplet).
                # psib12c oscillator gives spin-1/2 (conjugate partner of the gravitino).
                if hel_to_sign(row.get("psi1")) is not None and not is_plus_helicity(row.get("psi1")):
                    return ""
                return "3/2" if has_psib12 else "1/2"
            elif not left_st_ramond and right_st_ramond:
                # Left NS oscillator, right spacetime Ramond (h_R = ±½).
                # psi12c oscillator gives spin-1/2 (conjugate partner of the gravitino).
                if hel_to_sign(row.get("psib1")) is not None and not is_plus_helicity(row.get("psib1")):
                    return ""
                return "3/2" if has_psi12 else "1/2"
            else:
                # Both spacetime Ramond (8,8 sector).
                # Spin is derived from sigma (helicity product of spacetime Ramond vacua).
                # Vectors (sigma=+1 IIB / sigma=-1 IIA): count only psi1='+' (h=+1) states.
                # Scalars (sigma=-1 IIB / sigma=+1 IIA): count both psi1='+' and psi1='-'.
                if rr_gate(row, sec_vec):
                    sigma = sigma_from_psi_chi(row, sec_vec)
                    is_vector = (sigma == +1) if self.type_ii == "IIB" else (sigma == -1)
                    if is_vector:
                        if hel_to_sign(row.get("psi1")) is not None and not is_plus_helicity(row.get("psi1")):
                            return ""
                        return "1"
                    else:
                        return "0"
                if hel_to_sign(row.get("psi1")) is not None and not is_plus_helicity(row.get("psi1")):
                    return ""
                return "0"

        df["Spin"] = df.apply(compute_spin, axis=1)

        # Assign V_T to chi/chib oscillator states in (0,8)/(8,0) sectors where the
        # spacetime fermion psi/psib is periodic.  These are partners to the RS (spin-3/2)
        # state inside the same sector and form part of the RS supermultiplet.
        # This step is done AFTER compute_spin so Spin retains the physically correct value
        # (1/2 for these states, not 1).
        def assign_chi_vt_comment(row: pd.Series) -> str:
            comment = str(row.get("Comments", "")).strip()
            if comment:
                return comment          # never overwrite an existing label
            # Chi V_T only makes sense in TWISTED supersectors (RS multiplets
            # live in twisted sectors).  Untwisted sectors (1, S, Sbar, S+Sbar)
            # must not receive this label — their chi-oscillator states are
            # dilatino partners of the gravitino, not RS-multiplet partners.
            sk = row.get("super_key")
            if sk in untwisted_superkeys:
                return comment
            alpha_t = row.get("(α_L,α_R)")
            if alpha_t not in ((8, 0), (0, 8)):
                return comment
            st: Tuple[int, ...] = row["sector_tuple"]
            sec_vec = sector_vec_from_bsector(st)
            osc = oscillator_set(row)
            if alpha_t == (8, 0):
                if a_from_bc(sec_vec[0]) == 1:          # psi1 periodic → left spacetime Ramond
                    # Only label the primary (psi1 = '+') state; its complex conjugate
                    # (psi1 = '-') is not independently labelled.
                    if not is_plus_helicity(row.get("psi1")):
                        return comment
                    for k in range(1, self.comp_dim + 1, 2):
                        if f"chib{k}{k+1}" in osc:
                            return "V_T"
            else:  # (0, 8)
                if a_from_bc(sec_vec[left_end]) == 1:   # psib1 periodic → right spacetime Ramond
                    # Only label the primary (psib1 = '+') state; its complex conjugate
                    # (psib1 = '-') is not independently labelled.
                    if not is_plus_helicity(row.get("psib1")):
                        return comment
                    for k in range(1, self.comp_dim + 1, 2):
                        if f"chi{k}{k+1}" in osc:
                            return "V_T"
            return comment

        df["Comments"] = df.apply(assign_chi_vt_comment, axis=1)

        rs_superkeys = set(df.loc[df["is_rs_sector"] == True, "super_key"].tolist())

        # Suppress V_T/H_T that come from (8,8) twisted sectors inside RS supersectors.
        # Chi-oscillator V_T from (0,8)/(8,0) sectors must NOT be suppressed — they are
        # genuine members of the RS supermultiplet.
        #
        # Note for SUSY-broken models: rs_superkeys is built from rows where
        # is_rs_sector == True, i.e. from RS states that SURVIVED the GGSO
        # projections.  If a SUSY-breaking phase projects the RS multiplet out
        # entirely on a given side, the corresponding (8,0) or (0,8) RS states
        # are absent from df, so that supersector is not in rs_superkeys, and
        # the V_T/H_T states there are kept as independent surviving content.
        # The suppression rule therefore behaves correctly in both the SUSY
        # and SUSY-broken cases without modification.
        mask_rs_superkey_vh = (
            df["super_key"].isin(rs_superkeys) &
            df["Comments"].isin(["V_T", "H_T"]) &
            df["(α_L,α_R)"].apply(lambda x: x == (8, 8))
        )
        df.loc[mask_rs_superkey_vh, "Comments"] = ""
        # Spin is NOT reset here — V_T states retain their sigma-based spin="1" so that
        # the RS supersector spin breakdown correctly shows the vector content.

        def is_vh_t_sector_row(row: pd.Series) -> bool:
            st: Tuple[int, ...] = row["sector_tuple"]
            alpha_t = row["(α_L,α_R)"]
            if alpha_t != (8, 8):
                return False
            if is_rr_sector(st):
                return False
            if row["super_key"] in rs_superkeys:
                return False
            sec_vec = sector_vec_from_bsector(st)
            return rr_gate(row, sec_vec)

        df["is_vh_t_sector"] = df.apply(is_vh_t_sector_row, axis=1)

        def unique_keys_in_order(keys: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
            out: List[Tuple[int, ...]] = []
            seen = set()
            for k in keys:
                if k in untwisted_superkeys:
                    continue
                if k not in seen:
                    seen.add(k)
                    out.append(k)
            return out

        twisted_keys = unique_keys_in_order(df["super_key"].tolist())
        rs_keys = unique_keys_in_order(df.loc[df["is_rs_sector"] == True, "super_key"].tolist())
        vh_t_keys = unique_keys_in_order(df.loc[df["is_vh_t_sector"] == True, "super_key"].tolist())

        out = df[[
            "Sector",
            "State",
            "(α_L,α_R)",
            "Comments",
            "Spin",
            "super_key",
            "is_rs_sector",
            "is_vh_t_sector",
        ]].copy()
        return out, twisted_keys, rs_keys, vh_t_keys

    def spectrum_stats(
        self,
        processed_df: pd.DataFrame,
        twisted_keys: List[Tuple[int, ...]],
        rs_keys: List[Tuple[int, ...]],
        vh_t_keys: List[Tuple[int, ...]],
    ) -> SpectrumStats:
        mi_ok = bool(self.verify_basis_prod_matrix() and self.verify_gso_invariance())

        if processed_df is None or processed_df.empty:
            return SpectrumStats(
                mi_ok=mi_ok,
                n_susy=0,
                n_susy_L=0,
                n_susy_R=0,
                n_rs=0,
                n_rs_L=0,
                n_rs_R=0,
                n_v=0,
                n_h=0,
                n_v_rr=0,
                n_h_rr=0,
                n_v_t=0,
                n_h_t=0,
                spin_counts_total={"0": 0, "1/2": 0, "1": 0, "3/2": 0, "2": 0},
                n_rs_sectors=0,
                n_vh_t_sectors=0,
                twisted_supersectors=twisted_keys,
                rs_supersectors=rs_keys,
                vh_t_supersectors=vh_t_keys,
            )

        comments = processed_df["Comments"].fillna("").astype(str).str.strip().tolist()

        # SUSY counting: distinguish L and R gravitinos. Backwards-compatible "gravitino"
        # label (untagged) is treated as left-side by convention (it should not appear
        # in the new tagging path but is kept for safety).
        n_susy_L = sum(1 for c in comments if c.lower() in ("gravitino_l", "gravitino"))
        n_susy_R = sum(1 for c in comments if c.lower() == "gravitino_r")
        n_susy = n_susy_L + n_susy_R

        # RS counting: same convention. Side-resolved counts are useful when SUSY
        # is broken on one side, since the RS multiplet survives only on the side
        # whose supercharge is preserved.
        n_rs_L = sum(1 for c in comments if c in ("RS_L", "RS"))
        n_rs_R = sum(1 for c in comments if c == "RS_R")
        n_rs = n_rs_L + n_rs_R

        n_v_rr = sum(1 for c in comments if c == "V_RR")
        n_h_rr = sum(1 for c in comments if c == "H_RR")
        # Count twisted vectors/hypers from spin values in is_vh_t_sector rows.
        # These are (8,8) twisted non-RR sectors with both spacetime fermions periodic.
        # Counting from spin (set by the sigma computation in compute_spin) rather than
        # from the V_T/H_T label makes n_v_t/n_h_t independent of the Comments column.
        _vh_t = processed_df["is_vh_t_sector"] == True
        n_v_t = int((processed_df.loc[_vh_t, "Spin"] == "1").sum())
        n_h_t = int((processed_df.loc[_vh_t, "Spin"] == "0").sum())

        n_v = n_v_rr + n_v_t
        n_h = n_h_rr + n_h_t

        # Spin counts across the full spectrum -- the main observable for N=0 models.
        spin_series = processed_df["Spin"].fillna("").astype(str).str.strip() \
            if "Spin" in processed_df.columns else pd.Series([], dtype=str)
        spin_dict_raw = spin_series.value_counts().to_dict()
        spin_counts_total = {s: int(spin_dict_raw.get(s, 0)) for s in ("0", "1/2", "1", "3/2", "2")}

        rs_sector_df = processed_df[processed_df["is_rs_sector"] == True]
        vh_t_sector_df = processed_df[processed_df["is_vh_t_sector"] == True]

        n_rs_sectors = int(rs_sector_df["Sector"].astype(str).nunique())
        n_vh_t_sectors = int(vh_t_sector_df["Sector"].astype(str).nunique())

        return SpectrumStats(
            mi_ok=mi_ok,
            n_susy=int(n_susy),
            n_susy_L=int(n_susy_L),
            n_susy_R=int(n_susy_R),
            n_rs=int(n_rs),
            n_rs_L=int(n_rs_L),
            n_rs_R=int(n_rs_R),
            n_v=int(n_v),
            n_h=int(n_h),
            n_v_rr=int(n_v_rr),
            n_h_rr=int(n_h_rr),
            n_v_t=int(n_v_t),
            n_h_t=int(n_h_t),
            spin_counts_total=spin_counts_total,
            n_rs_sectors=n_rs_sectors,
            n_vh_t_sectors=n_vh_t_sectors,
            twisted_supersectors=twisted_keys,
            rs_supersectors=rs_keys,
            vh_t_supersectors=vh_t_keys,
        )

    def write_processed_csv(
        self,
        processed_df: pd.DataFrame,
        output_csv: str,
        twisted_keys: List[Tuple[int, ...]],
        rs_keys: List[Tuple[int, ...]],
        vh_t_keys: List[Tuple[int, ...]],
    ) -> None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

        # Determine the SUSY level (and side decomposition) from the gravitino counts
        # in the processed DF. Backward-compatible "gravitino" label is treated as
        # left-side by convention.
        comments_lower = (
            processed_df["Comments"].fillna("").astype(str).str.strip().str.lower()
        )
        n_susy_L: int = int(comments_lower.isin(["gravitino_l", "gravitino"]).sum())
        n_susy_R: int = int(comments_lower.eq("gravitino_r").sum())
        n_susy: int = n_susy_L + n_susy_R

        def key_to_heading(key_tuple: Tuple[int, ...]) -> str:
            display = [str(key_tuple[0]), "*", "*"] + [str(x) for x in key_tuple[1:]]
            return "[" + ", ".join(display) + "]"

        cols = ["Sector", "State", "(α_L,α_R)", "Comments", "Spin"]

        df = processed_df.copy()
        if df.empty:
            groups = [("Untwisted supersector:", df)]
        else:
            def parse_sector_str(s: str) -> Tuple[int, ...]:
                vals = s.strip("[]").split()
                return tuple(int(v) for v in vals)

            sector_tuples = df["Sector"].astype(str).apply(parse_sector_str)
            vec_len = len(sector_tuples.iloc[0])
            untwisted_bases = [
                (2,) + (0,) * (vec_len - 1),
                (0, 1, 0) + (0,) * (vec_len - 3),
                (0, 0, 1) + (0,) * (vec_len - 3),
                (0, 1, 1) + (0,) * (vec_len - 3),
            ]
            untwisted_set = set(untwisted_bases)
            untwisted_mask = sector_tuples.isin(untwisted_set)
            untwisted_df = df[untwisted_mask].copy()
            remaining_df = df[~untwisted_mask].copy()

            groups = [("Untwisted supersector:", untwisted_df)]
            seen = set()
            for k in remaining_df["super_key"].tolist():
                if k not in seen:
                    seen.add(k)
                    groups.append((k, remaining_df[remaining_df["super_key"] == k].copy()))

        with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)

            w.writerow(["Basis matrix: (psi,chi,y,w | psib,chib,yb,wb)"])
            for row_vals in self.basis:
                w.writerow([str(int(x)) if float(x).is_integer() else str(x) for x in row_vals])
            w.writerow([])

            w.writerow(["GGSO phase matrix:"])
            for row_vals in self.gso:
                w.writerow([str(int(np.real(x))) if np.isreal(x) else str(x) for x in row_vals])
            w.writerow([])

            sym = self.compute_internal_symmetry_groups()
            w.writerow(["Internal symmetry group:"])
            w.writerow([sym["sym_str"]])
            w.writerow(["Internal fermion groups (same BC in all basis vectors):"])
            w.writerow(["Group", "Left fermions", "SO_L", "Right fermions", "SO_R"])
            for g in sym["groups"]:
                lf_names = " ".join(n for n, _ in g["left_fermions"])  if g["left_fermions"]  else "(none)"
                rf_names = " ".join(n for n, _ in g["right_fermions"]) if g["right_fermions"] else "(none)"
                w.writerow([
                    str(g["signature"]),
                    lf_names,
                    g["so_left"]  or "-",
                    rf_names,
                    g["so_right"] or "-",
                ])
            w.writerow(["Complex-fermion pairings (LL / RR / LR):"])
            for na, _, nb, _, cname in sym["ll_pairs"]:
                w.writerow(["LL", f"({na},{nb})", f"-> {cname}"])
            for na, _, nb, _, cname in sym["rr_pairs"]:
                w.writerow(["RR", f"({na},{nb})", f"-> {cname}"])
            for na, _, nb, _, cname in sym["lr_pairs"]:
                w.writerow(["LR", f"({na},{nb})", f"-> {cname}"])
            if sym["left_singles"]:
                w.writerow(["L-singles"] + [n for n, _ in sym["left_singles"]])
            if sym["right_singles"]:
                w.writerow(["R-singles"] + [n for n, _ in sym["right_singles"]])
            w.writerow([])

            for heading, gdf in groups:
                if heading == "Untwisted supersector:":
                    w.writerow([heading] + [""] * (len(cols) - 1))
                else:
                    w.writerow([f"Supersector for base: {key_to_heading(heading)}"] + [""] * (len(cols) - 1))
                w.writerow(cols)
                if not gdf.empty:
                    for _, r in gdf.iterrows():
                        w.writerow([r["Sector"], r["State"], r["(α_L,α_R)"], r["Comments"], r.get("Spin", "")])

            w.writerow([])
            w.writerow(["RS producing supersectors:"])
            if rs_keys:
                w.writerow([""] + [key_to_heading(k) for k in rs_keys])
            else:
                w.writerow(["", "(none)"])

            w.writerow([])
            w.writerow(["V_T/H_T producing (twisted) supersectors:"])
            if vh_t_keys:
                w.writerow([""] + [key_to_heading(k) for k in vh_t_keys])
            else:
                w.writerow(["", "(none)"])

            w.writerow([])
            w.writerow(["All twisted supersectors:"])
            if twisted_keys:
                w.writerow([""] + [key_to_heading(k) for k in twisted_keys])
            else:
                w.writerow(["", "(none)"])

            # Spin breakdown tables for each supersector
            spin_order = ["0", "1/2", "1", "3/2", "2"]

            # Load multiplet library once — prefer CSV over computed fallback.
            # _multiplet_library handles any N >= 1; N=0 reports raw spin counts only.
            if n_susy >= 1:
                csv_lib = (
                    load_multiplet_library_csv(self.multiplet_csv, n_susy)
                    if self.multiplet_csv else None
                )
                full_mults = csv_lib if csv_lib is not None else _multiplet_library(n_susy)
                mult_source = "CSV" if csv_lib is not None else "computed"
            else:
                full_mults = []
                mult_source = ""

            w.writerow([])
            w.writerow(["Spin breakdown by supersector:"])
            # Report the L/R breakdown of SUSY so the reader can interpret broken
            # configurations correctly.
            w.writerow(["SUSY level (n_susy_L, n_susy_R, total):", n_susy_L, n_susy_R, n_susy])

            if n_susy >= 1:
                w.writerow(["Multiplet library source:", mult_source])
                w.writerow(["Multiplets used:"] + [
                    "%s=%s" % (nm, arr) for nm, arr in full_mults
                ])
            elif n_susy == 0:
                w.writerow([
                    "Multiplet library:",
                    "N=0 (non-SUSY): supermultiplet matching is not applicable. "
                    "Reporting spin counts only."
                ])

            for heading, gdf in groups:
                if heading == "Untwisted supersector:":
                    heading_str = "Untwisted supersector"
                else:
                    disp = [str(heading[0]), "*", "*"] + [str(x) for x in heading[1:]]
                    heading_str = "[" + " ".join(disp) + "]"

                w.writerow([])
                w.writerow([heading_str])
                w.writerow(["Spin", "Number of states"])

                spin_counts_d: Dict[str, int] = {}
                if gdf.empty or "Spin" not in gdf.columns:
                    for s in spin_order:
                        w.writerow([s, 0])
                else:
                    spin_counts_d = gdf["Spin"].value_counts().to_dict()
                    for s in spin_order:
                        w.writerow([s, spin_counts_d.get(s, 0)])

                # Build the observed spin count vector [n0, n1/2, n1, n3/2, n2]
                sc = [
                    spin_counts_d.get("0",   0),
                    spin_counts_d.get("1/2", 0),
                    spin_counts_d.get("1",   0),
                    spin_counts_d.get("3/2", 0),
                    spin_counts_d.get("2",   0),
                ]
                w.writerow(["Spin vector [n0, n1/2, n1, n3/2, n2]:", str(sc)])

                if n_susy >= 1:
                    # Match against the full multiplet library for every supersector,
                    # relying only on the supersector's own raw spin-state counts.
                    # SUGRA/RS terms automatically get a zero coefficient when the
                    # sector has no spin-2/spin-3/2 content, so there is no need to
                    # pre-filter the library using the RS_L/RS_R/V_T/H_T comment flags.
                    match_str = _match_multiplets_str(sc, full_mults)
                    w.writerow(["Supermultiplet matching (N=%d):" % n_susy, match_str])
                elif n_susy == 0:
                    # N=0: no supermultiplet matching. Report raw spin content.
                    w.writerow([
                        "Supermultiplet matching:",
                        "N=0 — no supermultiplets; spin counts above are the full content.",
                    ])

    def compute(
        self,
        *,
        write_raw_csv: Optional[str] = None,
        write_processed_csv: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, SpectrumStats]:
        failures = self._modular_invariance_diagnostics()
        if failures:
            lines = ["Modular invariance check FAILED — spectrum not computed."]
            lines += [f"  • {f}" for f in failures]
            raise ValueError("\n".join(lines))
        df_raw = self.massless_raw()
        if write_raw_csv:
            os.makedirs(os.path.dirname(write_raw_csv) or ".", exist_ok=True)
            df_raw.to_csv(write_raw_csv, index=False, encoding="utf-8-sig")

        df_proc, twisted, rs_keys, vh_t_keys = self.processed_from_raw(df_raw)
        stats = self.spectrum_stats(df_proc, twisted, rs_keys, vh_t_keys)

        if write_processed_csv:
            self.write_processed_csv(df_proc, write_processed_csv, twisted, rs_keys, vh_t_keys)

        return df_raw, df_proc, stats


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    BASIS_PATH = os.path.join(_here, "Input_typeII", "InBasis.txt")
    GSO_PATH   = os.path.join(_here, "Input_typeII", "InGSO.txt")

    out_dir = os.path.join(_here, "Output_typeII")
    os.makedirs(out_dir, exist_ok=True)

    MULT_CSV = os.path.join(_here, "Input_typeII", "supermultiplets.csv")
    ff = FreeFermionModel.from_files(BASIS_PATH, GSO_PATH, label="demo", type_ii="IIB", multiplet_csv=MULT_CSV)
    t0 = time.time()
    raw_path = os.path.join(out_dir, "massless_spectrum_output_raw.csv")
    proc_path = os.path.join(out_dir, "massless_spectrum_output_processed.csv")
    try:
        df_raw, df_proc, stats = ff.compute(write_raw_csv=raw_path, write_processed_csv=proc_path)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
    dt = time.time() - t0

    print(
        f"[ok] states={len(df_raw)} "
        f"susy=({stats.n_susy_L},{stats.n_susy_R}) total={stats.n_susy} "
        f"rs=({stats.n_rs_L},{stats.n_rs_R}) total={stats.n_rs} "
        f"rs_sec={stats.n_rs_sectors} vh_t_sec={stats.n_vh_t_sectors} "
        f"V={stats.n_v} H={stats.n_h} "
        f"(V_RR={stats.n_v_rr} H_RR={stats.n_h_rr} V_T={stats.n_v_t} H_T={stats.n_h_t}) "
        f"spins={stats.spin_counts_total} "
        f"({dt:.2f}s)"
    )
    print(f"[ok] wrote:\n  {raw_path}\n  {proc_path}")
