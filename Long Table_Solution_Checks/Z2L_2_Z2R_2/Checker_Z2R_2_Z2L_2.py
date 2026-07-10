# Z2R_2_Z2L_2 solutions checker for B1, B2, B3, B4
#
# Changes made:
# 1) Robust CSV loader for Stefan file that supports:
#    - Packed format with columns: index, n, m, k, l, N, M, K, L
#    - Packed format with columns: index, n12, m3456, k12, l3456, N, M, K, L
#    - Expanded format with columns: n1,n2,m3..m6,k1,k2,l3..l6,N1..N6,M1..M6,K1..K6,L1..L6
#    It also optionally keeps the first "index" column from Stefan's file.
#
# 2) Fix bug: the loop used "hand_solutions" but the variable is "stefan_solutions".
#
# 3) Output naming:
#    - Unique code solutions written to: Z2R_2_Z2L_2_unique_solutions.csv
#    - After mapping, write reverse-mapped unique code solutions to:
#      Z2L_2_Z2R_2_unique_solutions_mapped.csv
#      with a column "mapped_stefan_index" that is either a Stefan index (or a
#      comma-separated list if multiple Stefan rows map to the same code solution)
#      or "UNMATCHED".
#
# NOTE: Requires: z3-solver, pandas

from z3 import *
import pandas as pd
from multiprocessing import Pool, cpu_count
import ast


# ------------- Z3 MODEL BUILDING: B1, B2, B3, B4 ------------- #

def build_solver():
    # ----- Declare 0/1 integer variables -----
    # unbarred
    n1, n2 = Ints('n1 n2')
    N1, N2, N3, N4, N5, N6 = Ints('N1 N2 N3 N4 N5 N6')
    m3, m4, m5, m6 = Ints('m3 m4 m5 m6')
    M1, M2, M3, M4, M5, M6 = Ints('M1 M2 M3 M4 M5 M6')

    # barred
    k1, k2 = Ints('k1 k2')
    K1, K2, K3, K4, K5, K6 = Ints('K1 K2 K3 K4 K5 K6')
    l3, l4, l5, l6 = Ints('l3 l4 l5 l6')
    L1, L2, L3, L4, L5, L6 = Ints('L1 L2 L3 L4 L5 L6')

    all_bits = [
        n1, n2,
        N1, N2, N3, N4, N5, N6,
        m3, m4, m5, m6,
        M1, M2, M3, M4, M5, M6,
        k1, k2,
        K1, K2, K3, K4, K5, K6,
        l3, l4, l5, l6,
        L1, L2, L3, L4, L5, L6
    ]

    s = Solver()

    # All parameters are bits: 0 or 1
    for b in all_bits:
        s.add(Or(b == 0, b == 1))

    # ----- Gauge-fixing / basic restrictions ----- #

    def restrict_pair_00_10_11(a, b):
        """Restrict (a,b) to be in {(0,0), (1,0), (1,1)}."""
        return Or(
            And(a == 0, b == 0),
            And(a == 1, b == 0),
            And(a == 1, b == 1)
        )

    # n12, k12, m34, m56, l34, l56 ∈ {(00),(10),(11)}
    s.add(restrict_pair_00_10_11(n1, n2))
    s.add(restrict_pair_00_10_11(k1, k2))
    s.add(restrict_pair_00_10_11(m3, m4))
    s.add(restrict_pair_00_10_11(m5, m6))
    s.add(restrict_pair_00_10_11(l3, l4))
    s.add(restrict_pair_00_10_11(l5, l6))

    # ----- Canonical classes for (n12, m3456) (B1,B2 left) ----- #
    cfgL1 = And(n1 == 0, n2 == 0, m3 == 0, m4 == 0, m5 == 0, m6 == 0)
    cfgL2 = And(n1 == 0, n2 == 0, m3 == 1, m4 == 0, m5 == 1, m6 == 0)
    cfgL3 = And(n1 == 0, n2 == 0, m3 == 1, m4 == 1, m5 == 1, m6 == 1)
    cfgL4 = And(n1 == 1, n2 == 0, m3 == 1, m4 == 0, m5 == 1, m6 == 1)
    cfgL5 = And(n1 == 1, n2 == 1, m3 == 1, m4 == 1, m5 == 1, m6 == 1)
    cfgL6 = And(n1 == 1, n2 == 1, m3 == 1, m4 == 1, m5 == 0, m6 == 0)

    s.add(Or(cfgL1, cfgL2, cfgL3, cfgL4, cfgL5, cfgL6))

    # ----- Canonical classes for (k12, l3456) (B3,B4 right) ----- #
    cfgR1 = And(k1 == 0, k2 == 0, l3 == 0, l4 == 0, l5 == 0, l6 == 0)
    cfgR2 = And(k1 == 0, k2 == 0, l3 == 1, l4 == 0, l5 == 1, l6 == 0)
    cfgR3 = And(k1 == 0, k2 == 0, l3 == 1, l4 == 1, l5 == 1, l6 == 1)
    cfgR4 = And(k1 == 1, k2 == 0, l3 == 1, l4 == 0, l5 == 1, l6 == 1)
    cfgR5 = And(k1 == 1, k2 == 1, l3 == 1, l4 == 1, l5 == 1, l6 == 1)
    cfgR6 = And(k1 == 1, k2 == 1, l3 == 1, l4 == 1, l5 == 0, l6 == 0)

    s.add(Or(cfgR1, cfgR2, cfgR3, cfgR4, cfgR5, cfgR6))

    # "at most 3 entries in N, M, K, L are non-zero"
    N_vec = [N1, N2, N3, N4, N5, N6]
    M_vec = [M1, M2, M3, M4, M5, M6]
    K_vec = [K1, K2, K3, K4, K5, K6]
    L_vec = [L1, L2, L3, L4, L5, L6]

    s.add(Sum(N_vec) <= 3)
    s.add(Sum(M_vec) <= 3)
    s.add(Sum(K_vec) <= 3)
    s.add(Sum(L_vec) <= 3)

    # ----- Modular invariance constraints ----- #

    def dot(u, v):
        return Sum(ui * vi for ui, vi in zip(u, v))

    N = N_vec
    M = M_vec
    K = K_vec
    L = L_vec

    # "Squares" as sums of components
    n12_sq = n1 + n2
    m34_sq = m3 + m4
    m56_sq = m5 + m6
    N_sq = Sum(N)
    M_sq = Sum(M)
    k12_sq = k1 + k2
    l34_sq = l3 + l4
    l56_sq = l5 + l6
    K_sq = Sum(K)
    L_sq = Sum(L)

    # Hamming distance-style "square"
    NM_sq = Sum((Ni - Mi) * (Ni - Mi) for Ni, Mi in zip(N, M))
    KL_sq = Sum((Ki - Li) * (Ki - Li) for Ki, Li in zip(K, L))

    # Block 1: ModB1B2
    s.add((n12_sq - N_sq) % 4 == 0)
    s.add((m34_sq - M_sq) % 4 == 0)
    s.add((m56_sq - NM_sq) % 4 == 0)

    # Block 2: ModbarB1barB2
    s.add((k12_sq - K_sq) % 4 == 0)
    s.add((l34_sq - L_sq) % 4 == 0)
    s.add((l56_sq - KL_sq) % 4 == 0)

    # Sign vectors
    two_n12_minus_1 = [2 * n1 - 1, 2 * n2 - 1]
    two_k12_minus_1 = [2 * k1 - 1, 2 * k2 - 1]
    two_m34_minus_1 = [2 * m3 - 1, 2 * m4 - 1]
    two_l34_minus_1 = [2 * l3 - 1, 2 * l4 - 1]

    # Slices
    N12 = [N1, N2]
    N34 = [N3, N4]
    M12 = [M1, M2]
    M34 = [M3, M4]
    K12 = [K1, K2]
    K34 = [K3, K4]
    L12 = [L1, L2]
    L34 = [L3, L4]

    # Block 3: ModB1barB1
    lhs_B1barB1 = dot(two_n12_minus_1, K12) - dot(two_k12_minus_1, N12)
    rhs_B1barB1 = N_sq - K_sq
    s.add((lhs_B1barB1 - rhs_B1barB1) % 4 == 0)

    # Block 4: ModB2barB2
    lhs_B2barB2 = dot(two_m34_minus_1, L34) - dot(two_l34_minus_1, M34)
    rhs_B2barB2 = M_sq - L_sq
    s.add((lhs_B2barB2 - rhs_B2barB2) % 4 == 0)

    # Block 5: ModB1barB2
    lhs_B1barB2 = dot(two_n12_minus_1, L12) - dot(two_l34_minus_1, N34)
    rhs_B1barB2 = N_sq - L_sq
    s.add((lhs_B1barB2 - rhs_B1barB2) % 4 == 0)

    # Block 6: ModB2barB1
    lhs_B2barB1 = dot(two_m34_minus_1, K34) - dot(two_k12_minus_1, M12)
    rhs_B2barB1 = M_sq - K_sq
    s.add((lhs_B2barB1 - rhs_B2barB1) % 4 == 0)

    # ----- Build full 40-component vectors for E, B1, B2, B3, B4 ----- #

    # E:
    E_left = [0] * 8 + [1] * 6 + [1] * 6
    E_right = [0] * 8 + [1] * 6 + [1] * 6

    # B1:
    B1_left = [0, 0, 0, 0, 1, 1, 1, 1,
               n1, n2, 1, 1, 1, 1,
               n1, n2, 0, 0, 0, 0]
    B1_right = [0] * 8 + N + N

    # B2:
    B2_left = [0, 0, 1, 1, 0, 0, 1, 1,
               1, 1, m3, m4, 1 - m5, 1 - m6,
               0, 0, m3, m4, m5, m6]
    B2_right = [0] * 8 + M + M

    # B3:
    B3_left = [0] * 8 + K + K
    B3_right = [0, 0, 0, 0, 1, 1, 1, 1,
                k1, k2, 1, 1, 1, 1,
                k1, k2, 0, 0, 0, 0]

    # B4:
    B4_left = [0] * 8 + L + L
    B4_right = [0, 0, 1, 1, 0, 0, 1, 1,
                1, 1, l3, l4, 1 - l5, 1 - l6,
                0, 0, l3, l4, l5, l6]

    def lorentz_parity_constraint(vA_left, vA_right,
                                  vB_left, vB_right,
                                  vC_left, vC_right,
                                  vD_left, vD_right):
        left_terms = [vA_left[i] * vB_left[i] * vC_left[i] * vD_left[i] for i in range(len(vA_left))]
        right_terms = [vA_right[i] * vB_right[i] * vC_right[i] * vD_right[i] for i in range(len(vA_right))]
        return (Sum(left_terms) - Sum(right_terms)) % 2 == 0

    # 5 parity constraints:
    s.add(lorentz_parity_constraint(E_left, E_right, B1_left, B1_right, B2_left, B2_right, B3_left, B3_right))
    s.add(lorentz_parity_constraint(E_left, E_right, B1_left, B1_right, B2_left, B2_right, B4_left, B4_right))
    s.add(lorentz_parity_constraint(E_left, E_right, B1_left, B1_right, B3_left, B3_right, B4_left, B4_right))
    s.add(lorentz_parity_constraint(E_left, E_right, B2_left, B2_right, B3_left, B3_right, B4_left, B4_right))
    s.add(lorentz_parity_constraint(B1_left, B1_right, B2_left, B2_right, B3_left, B3_right, B4_left, B4_right))

    return s, all_bits


# ------------- ENUMERATION OF ALL SOLUTIONS ------------- #

def enumerate_solutions():
    s, vars_ = build_solver()
    sols = []

    while s.check() == sat:
        m = s.model()
        sol = {str(v): m[v].as_long() for v in vars_}
        sols.append(sol)

        # Block this exact solution:
        s.add(Or(*[v != m[v] for v in vars_]))

    return sols


# ------------- EQUIVALENCE KEY (LEFT & RIGHT PAIRWISE) ------------- #

def equivalence_key(sol):
    """
    Equivalence for the B1,B2,B3,B4 case.
    (unchanged)
    """

    def vec(prefix, length):
        return [sol[f"{prefix}{i}"] for i in range(1, length + 1)]

    n1, n2 = sol["n1"], sol["n2"]
    m3, m4, m5, m6 = sol["m3"], sol["m4"], sol["m5"], sol["m6"]
    k1, k2 = sol["k1"], sol["k2"]
    l3, l4, l5, l6 = sol["l3"], sol["l4"], sol["l5"], sol["l6"]

    N = vec("N", 6)
    M = vec("M", 6)
    K = vec("K", 6)
    L = vec("L", 6)

    # ---------- LEFT equivalences ----------
    if n1 == n2:
        orig = (K[0], K[1], L[0], L[1])
        swapped = (K[1], K[0], L[1], L[0])
        if swapped < orig:
            K[0], K[1], L[0], L[1] = swapped

    if m3 == m4:
        orig = (K[2], K[3], L[2], L[3])
        swapped = (K[3], K[2], L[3], L[2])
        if swapped < orig:
            K[2], K[3], L[2], L[3] = swapped

    if m5 == m6:
        orig = (K[4], K[5], L[4], L[5])
        swapped = (K[5], K[4], L[5], L[4])
        if swapped < orig:
            K[4], K[5], L[4], L[5] = swapped

    # ---------- RIGHT equivalences ----------
    if k1 == k2:
        orig = (N[0], N[1], M[0], M[1])
        swapped = (N[1], N[0], M[1], M[0])
        if swapped < orig:
            N[0], N[1], M[0], M[1] = swapped

    if l3 == l4:
        orig = (N[2], N[3], M[2], M[3])
        swapped = (N[3], N[2], M[3], M[2])
        if swapped < orig:
            N[2], N[3], M[2], M[3] = swapped

    if l5 == l6:
        orig = (N[4], N[5], M[4], M[5])
        swapped = (N[5], N[4], M[5], M[4])
        if swapped < orig:
            N[4], N[5], M[4], M[5] = swapped

    # ---------- Build key ----------
    n12 = (n1, n2)
    m3456 = (m3, m4, m5, m6)
    k12 = (k1, k2)
    l3456 = (l3, l4, l5, l6)

    key = (n12, m3456, k12, l3456, tuple(N), tuple(M), tuple(K), tuple(L))
    return key


def pack_solution(sol):
    """Return packed row for CSV (8 tuple-columns)."""
    return {
        "n12": (sol["n1"], sol["n2"]),
        "m3456": (sol["m3"], sol["m4"], sol["m5"], sol["m6"]),
        "k12": (sol["k1"], sol["k2"]),
        "l3456": (sol["l3"], sol["l4"], sol["l5"], sol["l6"]),
        "N": tuple(sol[f"N{i}"] for i in range(1, 7)),
        "M": tuple(sol[f"M{i}"] for i in range(1, 7)),
        "K": tuple(sol[f"K{i}"] for i in range(1, 7)),
        "L": tuple(sol[f"L{i}"] for i in range(1, 7)),
    }


# ------------- PARALLEL POST-PROCESSING TO UNIQUE REPRESENTATIVES ------------- #

def _compute_key_for_solution(args):
    idx, sol = args
    return idx, equivalence_key(sol)


def compute_keys_parallel(solutions, n_procs=None):
    if n_procs is None:
        n_procs = max(1, cpu_count() - 1)

    if n_procs == 1 or len(solutions) == 0:
        return [(i, equivalence_key(sol)) for i, sol in enumerate(solutions)]

    with Pool(processes=n_procs) as pool:
        results = pool.map(_compute_key_for_solution, list(enumerate(solutions)))
    return results


def get_unique_representatives(solutions, parallel=True):
    if not solutions:
        return []

    if parallel:
        idx_key_list = compute_keys_parallel(solutions)
    else:
        idx_key_list = [(i, equivalence_key(sol)) for i, sol in enumerate(solutions)]

    key_to_idx = {}
    for idx, key in idx_key_list:
        if key not in key_to_idx:
            key_to_idx[key] = idx

    unique_solutions = [solutions[i] for i in sorted(key_to_idx.values())]
    return unique_solutions


# ------------- MODULAR INVARIANCE CHECK (WITH DIAGNOSTICS) ------------- #

def check_modular_invariance_B1B2B3B4(sol):
    """
    (unchanged, except uses K/L naming consistently)
    """
    n1, n2 = sol["n1"], sol["n2"]
    m3, m4, m5, m6 = sol["m3"], sol["m4"], sol["m5"], sol["m6"]
    k1, k2 = sol["k1"], sol["k2"]
    l3, l4, l5, l6 = sol["l3"], sol["l4"], sol["l5"], sol["l6"]

    N = [sol[f"N{i}"] for i in range(1, 7)]
    M = [sol[f"M{i}"] for i in range(1, 7)]
    K = [sol[f"K{i}"] for i in range(1, 7)]
    L = [sol[f"L{i}"] for i in range(1, 7)]

    n12_sq = n1 + n2
    m34_sq = m3 + m4
    m56_sq = m5 + m6
    N_sq = sum(N)
    M_sq = sum(M)
    k12_sq = k1 + k2
    l34_sq = l3 + l4
    l56_sq = l5 + l6
    K_sq = sum(K)
    L_sq = sum(L)

    NM_sq = sum((Ni - Mi) ** 2 for Ni, Mi in zip(N, M))
    KL_sq = sum((Ki - Li) ** 2 for Ki, Li in zip(K, L))

    c1 = ((n12_sq - N_sq) % 4 == 0)
    c2 = ((m34_sq - M_sq) % 4 == 0)
    c3 = ((m56_sq - NM_sq) % 4 == 0)
    c4 = ((k12_sq - K_sq) % 4 == 0)
    c5 = ((l34_sq - L_sq) % 4 == 0)
    c6 = ((l56_sq - KL_sq) % 4 == 0)

    two_n12_minus_1 = [2 * n1 - 1, 2 * n2 - 1]
    two_k12_minus_1 = [2 * k1 - 1, 2 * k2 - 1]
    two_m34_minus_1 = [2 * m3 - 1, 2 * m4 - 1]
    two_l34_minus_1 = [2 * l3 - 1, 2 * l4 - 1]

    N12 = N[:2]
    N34 = N[2:4]
    M12 = M[:2]
    M34 = M[2:4]
    K12 = K[:2]
    K34 = K[2:4]
    L12 = L[:2]
    L34 = L[2:4]

    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    lhs_B1barB1 = dot(two_n12_minus_1, K12) - dot(two_k12_minus_1, N12)
    rhs_B1barB1 = N_sq - K_sq
    c7 = ((lhs_B1barB1 - rhs_B1barB1) % 4 == 0)

    lhs_B2barB2 = dot(two_m34_minus_1, L34) - dot(two_l34_minus_1, M34)
    rhs_B2barB2 = M_sq - L_sq
    c8 = ((lhs_B2barB2 - rhs_B2barB2) % 4 == 0)

    lhs_B1barB2 = dot(two_n12_minus_1, L12) - dot(two_l34_minus_1, N34)
    rhs_B1barB2 = N_sq - L_sq
    c9 = ((lhs_B1barB2 - rhs_B1barB2) % 4 == 0)

    lhs_B2barB1 = dot(two_m34_minus_1, K34) - dot(two_k12_minus_1, M12)
    rhs_B2barB1 = M_sq - K_sq
    c10 = ((lhs_B2barB1 - rhs_B2barB1) % 4 == 0)

    # Parity constraints (same structure)
    E_left = [0] * 8 + [1] * 6 + [1] * 6
    E_right = [0] * 8 + [1] * 6 + [1] * 6

    B1_left = [0, 0, 0, 0, 1, 1, 1, 1,
               n1, n2, 1, 1, 1, 1,
               n1, n2, 0, 0, 0, 0]
    B1_right = [0] * 8 + N + N

    B2_left = [0, 0, 1, 1, 0, 0, 1, 1,
               1, 1, m3, m4, 1 - m5, 1 - m6,
               0, 0, m3, m4, m5, m6]
    B2_right = [0] * 8 + M + M

    B3_left = [0] * 8 + K + K
    B3_right = [0, 0, 0, 0, 1, 1, 1, 1,
                k1, k2, 1, 1, 1, 1,
                k1, k2, 0, 0, 0, 0]

    B4_left = [0] * 8 + L + L
    B4_right = [0, 0, 1, 1, 0, 0, 1, 1,
                1, 1, l3, l4, 1 - l5, 1 - l6,
                0, 0, l3, l4, l5, l6]

    def parity_ok(A_left, A_right, B_left, B_right, C_left, C_right, D_left, D_right):
        left_terms = [A_left[i] * B_left[i] * C_left[i] * D_left[i] for i in range(len(A_left))]
        right_terms = [A_right[i] * B_right[i] * C_right[i] * D_right[i] for i in range(len(A_right))]
        return ((sum(left_terms) - sum(right_terms)) % 2) == 0

    c11 = parity_ok(E_left, E_right, B1_left, B1_right, B2_left, B2_right, B3_left, B3_right)
    c12 = parity_ok(E_left, E_right, B1_left, B1_right, B2_left, B2_right, B4_left, B4_right)
    c13 = parity_ok(E_left, E_right, B1_left, B1_right, B3_left, B3_right, B4_left, B4_right)
    c14 = parity_ok(E_left, E_right, B2_left, B2_right, B3_left, B3_right, B4_left, B4_right)
    c15 = parity_ok(B1_left, B1_right, B2_left, B2_right, B3_left, B3_right, B4_left, B4_right)

    constraints = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]
    names = [
        "n12_vs_N",
        "m34_vs_M",
        "m56_vs_N_minus_M",
        "k12_vs_K",
        "l34_vs_L",
        "l56_vs_K_minus_L",
        "B1barB1",
        "B2barB2",
        "B1barB2",
        "B2barB1",
        "E_B1_B2_B3_parity",
        "E_B1_B2_B4_parity",
        "E_B1_B3_B4_parity",
        "E_B2_B3_B4_parity",
        "B1_B2_B3_B4_parity",
    ]

    failed = [names[i] for i, ok in enumerate(constraints) if not ok]
    return all(constraints), failed


# ------------- LOAD STEFAN's SOLUTIONS ------------- #

def _parse_tuple_cell(x, expected_len, colname):
    """
    Parse a cell like "(0, 1, 0)" into a tuple of ints.
    Accepts tuples/lists already.
    """
    if isinstance(x, (tuple, list)):
        t = tuple(int(v) for v in x)
    else:
        if pd.isna(x):
            raise ValueError(f"Missing value in column '{colname}'.")
        s = str(x).strip()

        # If someone wrote "0,1" without parentheses, tolerate it
        if s and s[0] != "(" and "," in s:
            s = f"({s})"

        try:
            t = ast.literal_eval(s)
        except Exception as e:
            raise ValueError(f"Could not parse tuple in column '{colname}': {x!r}") from e

        if not isinstance(t, (tuple, list)):
            raise ValueError(f"Column '{colname}' is not a tuple/list: {x!r}")
        t = tuple(int(v) for v in t)

    if len(t) != expected_len:
        raise ValueError(f"Column '{colname}' expected length {expected_len}, got {len(t)}: {t!r}")
    return t


def load_stefan_solutions_from_csv(filename="Z2L_2_Z2R_2-reduced_reformatted.csv", keep_row_index=True):
    """
    Supports:

    A1) Packed (your reformatted file):
        columns: index, n, m, k, l, N, M, K, L
        where:
          n has 2 comps, m has 4 comps, k has 2 comps, l has 4 comps,
          N,M,K,L have 6 comps.

    A2) Packed (older naming):
        columns: index, n12, m3456, k12, l3456, N, M, K, L

    B) Expanded:
        columns: n1,n2,m3,m4,m5,m6,k1,k2,l3,l4,l5,l6,N1..N6,M1..M6,K1..K6,L1..L6

    Returns: list[dict] with keys n1,n2,m3..m6,k1,k2,l3..l6,N1..N6,M1..M6,K1..K6,L1..L6
             plus optional "_stefan_index" if keep_row_index=True and index col exists.
    """
    df = pd.read_csv(filename)
    cols = set(df.columns)

    # Identify an optional Stefan index column
    idx_col = None
    for candidate in ["index", "idx", "row_index", "stefan_index"]:
        if candidate in cols:
            idx_col = candidate
            break

    def attach_index(sol_dict, row):
        if keep_row_index and idx_col is not None:
            try:
                sol_dict["_stefan_index"] = int(row[idx_col])
            except Exception:
                sol_dict["_stefan_index"] = row[idx_col]
        return sol_dict

    # ---- Packed format (new): index, n,m,k,l,N,M,K,L ----
    if {"n", "m", "k", "l", "N", "M", "K", "L"}.issubset(cols):
        sols = []
        for row in df.itertuples(index=False):
            rowd = row._asdict()

            n12 = _parse_tuple_cell(rowd["n"], 2, "n")
            m3456 = _parse_tuple_cell(rowd["m"], 4, "m")
            k12 = _parse_tuple_cell(rowd["k"], 2, "k")
            l3456 = _parse_tuple_cell(rowd["l"], 4, "l")
            N = _parse_tuple_cell(rowd["N"], 6, "N")
            M = _parse_tuple_cell(rowd["M"], 6, "M")
            K = _parse_tuple_cell(rowd["K"], 6, "K")
            L = _parse_tuple_cell(rowd["L"], 6, "L")

            sol = {
                "n1": n12[0], "n2": n12[1],
                "m3": m3456[0], "m4": m3456[1], "m5": m3456[2], "m6": m3456[3],
                "k1": k12[0], "k2": k12[1],
                "l3": l3456[0], "l4": l3456[1], "l5": l3456[2], "l6": l3456[3],
            }
            sol.update({f"N{i}": N[i - 1] for i in range(1, 7)})
            sol.update({f"M{i}": M[i - 1] for i in range(1, 7)})
            sol.update({f"K{i}": K[i - 1] for i in range(1, 7)})
            sol.update({f"L{i}": L[i - 1] for i in range(1, 7)})

            sol = attach_index(sol, rowd)
            sols.append(sol)
        return sols

    # ---- Packed format (older names): n12,m3456,k12,l3456,N,M,K,L ----
    if {"n12", "m3456", "k12", "l3456", "N", "M", "K", "L"}.issubset(cols):
        sols = []
        for row in df.itertuples(index=False):
            rowd = row._asdict()

            n12 = _parse_tuple_cell(rowd["n12"], 2, "n12")
            m3456 = _parse_tuple_cell(rowd["m3456"], 4, "m3456")
            k12 = _parse_tuple_cell(rowd["k12"], 2, "k12")
            l3456 = _parse_tuple_cell(rowd["l3456"], 4, "l3456")
            N = _parse_tuple_cell(rowd["N"], 6, "N")
            M = _parse_tuple_cell(rowd["M"], 6, "M")
            K = _parse_tuple_cell(rowd["K"], 6, "K")
            L = _parse_tuple_cell(rowd["L"], 6, "L")

            sol = {
                "n1": n12[0], "n2": n12[1],
                "m3": m3456[0], "m4": m3456[1], "m5": m3456[2], "m6": m3456[3],
                "k1": k12[0], "k2": k12[1],
                "l3": l3456[0], "l4": l3456[1], "l5": l3456[2], "l6": l3456[3],
            }
            sol.update({f"N{i}": N[i - 1] for i in range(1, 7)})
            sol.update({f"M{i}": M[i - 1] for i in range(1, 7)})
            sol.update({f"K{i}": K[i - 1] for i in range(1, 7)})
            sol.update({f"L{i}": L[i - 1] for i in range(1, 7)})

            sol = attach_index(sol, rowd)
            sols.append(sol)
        return sols

    # ---- Expanded format ----
    required = (
        ["n1", "n2", "m3", "m4", "m5", "m6", "k1", "k2", "l3", "l4", "l5", "l6"]
        + [f"N{i}" for i in range(1, 7)]
        + [f"M{i}" for i in range(1, 7)]
        + [f"K{i}" for i in range(1, 7)]
        + [f"L{i}" for i in range(1, 7)]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Unrecognized stefan CSV format. Missing columns:\n  "
            + ", ".join(missing)
            + "\nExpected packed columns (n,m,k,l,N,M,K,L) or (n12,m3456,k12,l3456,N,M,K,L) "
              "or expanded columns."
        )

    sols = []
    for row in df.itertuples(index=False):
        rowd = row._asdict()
        sol = {c: int(rowd[c]) for c in required}
        sol = attach_index(sol, rowd)
        sols.append(sol)
    return sols


# ------------- MAIN ------------- #

if __name__ == "__main__":
    # 1. Enumerate all Z3 models under the full constraints
    print("Enumerating all Z3 solutions (B1,B2,B3,B4)...")
    all_solutions = enumerate_solutions()
    print(f"Total raw solutions found: {len(all_solutions)}")

    # 2. Reduce to unique representatives under equivalence
    print("Computing unique representatives under equivalence...")
    unique_solutions = get_unique_representatives(all_solutions, parallel=True)
    print(f"Unique solutions after quotienting: {len(unique_solutions)}")

    # 3. Pack into 8-column rows and save
    rows = [pack_solution(sol) for sol in unique_solutions]
    df_unique = pd.DataFrame(rows, columns=[
        "n12", "m3456", "k12", "l3456", "N", "M", "K", "L"
    ])
    out_name = "Z2R_2_Z2L_2_unique_solutions.csv"
    df_unique.to_csv(out_name, index=False)
    print(f"Saved unique solutions to {out_name}")

    # 4. Load Stefan solutions (your reformatted file)
    stefan_file = "Z2L_2_Z2R_2-reduced_reformatted.csv"
    print(f"\nLoading stefan solutions CSV ({stefan_file})...")
    stefan_solutions = load_stefan_solutions_from_csv(stefan_file, keep_row_index=True)
    print(f"Loaded {len(stefan_solutions)} stefan solutions")

    # 5. Build map from equivalence key -> index in unique_solutions (1-based)
    key_to_code_idx = {equivalence_key(sol): i + 1 for i, sol in enumerate(unique_solutions)}

    # 6. For each Stefan solution, check modular invariance and equivalence class
    mapped_rows = []
    for sol in stefan_solutions:
        n12 = (sol["n1"], sol["n2"])
        m3456 = (sol["m3"], sol["m4"], sol["m5"], sol["m6"])
        k12 = (sol["k1"], sol["k2"])
        l3456 = (sol["l3"], sol["l4"], sol["l5"], sol["l6"])
        N_vec = tuple(sol[f"N{i}"] for i in range(1, 7))
        M_vec = tuple(sol[f"M{i}"] for i in range(1, 7))
        K_vec = tuple(sol[f"K{i}"] for i in range(1, 7))
        L_vec = tuple(sol[f"L{i}"] for i in range(1, 7))

        row = {
            "stefan_index": sol.get("_stefan_index", None),
            "n12": n12,
            "m3456": m3456,
            "k12": k12,
            "l3456": l3456,
            "N": N_vec,
            "M": M_vec,
            "K": K_vec,
            "L": L_vec,
        }

        ok, failed = check_modular_invariance_B1B2B3B4(sol)
        row["modular_invariant"] = ok
        row["failed_constraints"] = ",".join(failed) if not ok else ""

        key = equivalence_key(sol)
        row["equivalent_to_code_solution"] = key_to_code_idx.get(key, -1)  # -1 = no match

        mapped_rows.append(row)

    df_mapped = pd.DataFrame(mapped_rows)
    out_name_mapped = "B1B2B3B4_stefan_mapped.csv"
    df_mapped.to_csv(out_name_mapped, index=False)
    print(f"Saved stefan solution mapping to {out_name_mapped}")

    # 7. Reverse mapping: for each unique code solution, list which Stefan indices map to it
    code_key_to_stefan_indices = {}
    for sol in stefan_solutions:
        key = equivalence_key(sol)
        stefan_idx = sol.get("_stefan_index", None)
        # If the file had no index column, fall back to None and we’ll still record a row number later
        if stefan_idx is None:
            continue
        code_idx = key_to_code_idx.get(key, None)
        if code_idx is None:
            continue
        code_key_to_stefan_indices.setdefault(key, []).append(stefan_idx)

    # Build the mapped unique code solutions output
    mapped_code_rows = []
    for i, sol in enumerate(unique_solutions, start=1):
        key = equivalence_key(sol)
        stefan_list = code_key_to_stefan_indices.get(key, [])
        mapped_code_rows.append({
            "code_solution_index": i,
            **pack_solution(sol),
            "mapped_stefan_index": "UNMATCHED" if not stefan_list else ",".join(str(x) for x in stefan_list)
        })

    df_unique_mapped = pd.DataFrame(
        mapped_code_rows,
        columns=["code_solution_index", "n12", "m3456", "k12", "l3456", "N", "M", "K", "L", "mapped_stefan_index"]
    )
    out_unique_mapped = "Z2L_2_Z2R_2_unique_solutions_mapped.csv"
    df_unique_mapped.to_csv(out_unique_mapped, index=False)
    print(f"Saved reverse-mapped unique code solutions to {out_unique_mapped}")
