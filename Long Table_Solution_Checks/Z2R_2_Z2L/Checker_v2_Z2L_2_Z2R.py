# Z2L_2_Z2R solutions checker for B1, B2, B3
#
# - Enumerates all Z3 solutions with full modular constraints
#   and E·B1·B2·B3 parity, plus canonical (n12, m3456) classes.
# - Quotients by equivalence under allowed relabellings.
# - Loads your 21 stefan solutions from Z2L_2_Z2R-reduced_stefan_formatted.csv
#   (tuple-packed columns: n12,m3456,k12,N,M,K),
#   checks modular invariance, and maps each to a unique solution.
# - Counts the number of solutions in each of the 18 (k12)×(n12,m3456) configurations
#   and compares against your expected table.
# - Produces mapping diagnostics (sorted mapped indices, missing indices, duplicates, collisions).
# - Writes:
#     1) Z2L_2_Z2R_unique_code_solutions.csv
#     2) stefan_reduced_mapped.csv
#     3) Z2L_2_Z2R_unique_code_solutions_mapped.csv  (reverse mapping code->stefan)

from z3 import *
import pandas as pd
import ast


# ------------- Z3 MODEL BUILDING FOR B1, B2, B3 ------------- #

def build_solver():
    # ----- Declare 0/1 integer variables -----
    # unbarred (B1, B2 side)
    n1, n2 = Ints('n1 n2')
    N1, N2, N3, N4, N5, N6 = Ints('N1 N2 N3 N4 N5 N6')
    m3, m4, m5, m6 = Ints('m3 m4 m5 m6')
    M1, M2, M3, M4, M5, M6 = Ints('M1 M2 M3 M4 M5 M6')

    # barred B1 only (B3)
    k1, k2 = Ints('k1 k2')
    K1, K2, K3, K4, K5, K6 = Ints('K1 K2 K3 K4 K5 K6')

    all_bits = [
        n1, n2,
        N1, N2, N3, N4, N5, N6,
        m3, m4, m5, m6,
        M1, M2, M3, M4, M5, M6,
        k1, k2,
        K1, K2, K3, K4, K5, K6,
    ]

    s = Solver()

    # All parameters 0 or 1
    for b in all_bits:
        s.add(Or(b == 0, b == 1))

    # ----- n12,k12,m34,m56 = 00,10,11 ----- #

    def restrict_pair_00_10_11(a, b):
        return Or(
            And(a == 0, b == 0),
            And(a == 1, b == 0),
            And(a == 1, b == 1),
        )

    s.add(restrict_pair_00_10_11(n1, n2))
    s.add(restrict_pair_00_10_11(k1, k2))
    s.add(restrict_pair_00_10_11(m3, m4))
    s.add(restrict_pair_00_10_11(m5, m6))

    # ---- Restrict (n12, m3456) to the 6 Z2R^2 B1,B2 classes ----
    c1 = And(n1 == 0, n2 == 0, m3 == 0, m4 == 0, m5 == 0, m6 == 0)
    c2 = And(n1 == 0, n2 == 0, m3 == 1, m4 == 0, m5 == 1, m6 == 0)
    c3 = And(n1 == 0, n2 == 0, m3 == 1, m4 == 1, m5 == 1, m6 == 1)
    c4 = And(n1 == 1, n2 == 0, m3 == 1, m4 == 0, m5 == 1, m6 == 1)
    c5 = And(n1 == 1, n2 == 1, m3 == 1, m4 == 1, m5 == 1, m6 == 1)
    c6 = And(n1 == 1, n2 == 1, m3 == 1, m4 == 1, m5 == 0, m6 == 0)
    s.add(Or(c1, c2, c3, c4, c5, c6))

    # at most 3 entries non-zero in N, M, K
    N = [N1, N2, N3, N4, N5, N6]
    M = [M1, M2, M3, M4, M5, M6]
    K = [K1, K2, K3, K4, K5, K6]

    s.add(Sum(N) <= 3)
    s.add(Sum(M) <= 3)
    s.add(Sum(K) <= 3)

    # ----- Modular invariance constraints (B1,B2,B3) ----- #

    n12_sq = n1 + n2
    m34_sq = m3 + m4
    m56_sq = m5 + m6
    N_sq = Sum(N)
    M_sq = Sum(M)
    k12_sq = k1 + k2
    K_sq = Sum(K)

    # Hamming distance-style "square"
    NM_sq = Sum((Ni - Mi) * (Ni - Mi) for Ni, Mi in zip(N, M))

    # B1B1,B2B2, B1B2
    s.add((n12_sq - N_sq) % 4 == 0)
    s.add((m34_sq - M_sq) % 4 == 0)
    s.add((m56_sq - NM_sq) % 4 == 0)

    # k12 / K constraint
    s.add((k12_sq - K_sq) % 4 == 0)

    # B1 B1b
    lhs = 2 * (n1 * K1 + n2 * K2) + K3 + K4 + K5 + K6
    rhs = 2 * (k1 * N1 + k2 * N2) + N3 + N4 + N5 + N6
    s.add((lhs - rhs) % 4 == 0)

    # B2 B1b
    lhs_2_1b = K1 + K2 + 2 * (m3 * K3 + m4 * K4) + K5 + K6
    rhs_2_1b = 2 * (k1 * M1 + k2 * M2) + M3 + M4 + M5 + M6
    s.add((lhs_2_1b - rhs_2_1b) % 4 == 0)

    # ----- Additional (E·)B1·B2·B3 constraint -----
    lhs_1_2_1b = n1 * K1 + n2 * K2 + m3 * K3 + m4 * K4 + (1 - m5) * K5 + (1 - m6) * K6
    rhs_1_2_1b = 2 * (N1 * M1 * k1 + N2 * M2 * k2) + N3 * M3 + N4 * M4 + N5 * M5 + N6 * M6
    s.add((lhs_1_2_1b - rhs_1_2_1b) % 2 == 0)

    return s, all_bits


# ------------- ENUMERATION OF ALL SOLUTIONS ------------- #

def enumerate_solutions():
    s, vars_ = build_solver()
    sols = []

    while s.check() == sat:
        m = s.model()
        sol = {str(v): m[v].as_long() for v in vars_}
        sols.append(sol)
        s.add(Or(*[v != m[v] for v in vars_]))  # block exact model

    return sols


# ------------- EQUIVALENCE KEY ------------- #

def equivalence_key(sol):
    """
    Equivalence for the B1,B2,B3 case, based on allowed relabellings of
    X^1..6 (left) and X'^1..6 (right).

    LEFT (X^1..X^6):
      (1,2): if n1 == n2, we can swap K1,K2.
      (3,4): if m3 == m4, we can swap K3,K4.
      (5,6): if m5 == m6, we can swap K5,K6.

    RIGHT (X'^1..X'^6):
      (1',2'): if k1 == k2, we can swap (N1,N2; M1,M2).
      (3',4',5',6'): use only invariants:
          Nrest_sq  = sum N3..N6
          Mrest_sq  = sum M3..M6
          both_rest = # of i in {3..6} with (N_i,M_i)=(1,1).
    """
    n1, n2 = sol["n1"], sol["n2"]
    k1, k2 = sol["k1"], sol["k2"]
    m3, m4, m5, m6 = sol["m3"], sol["m4"], sol["m5"], sol["m6"]
    N1, N2, N3, N4, N5, N6 = (sol["N1"], sol["N2"], sol["N3"], sol["N4"], sol["N5"], sol["N6"])
    M1, M2, M3, M4, M5, M6 = (sol["M1"], sol["M2"], sol["M3"], sol["M4"], sol["M5"], sol["M6"])
    K1, K2, K3, K4, K5, K6 = (sol["K1"], sol["K2"], sol["K3"], sol["K4"], sol["K5"], sol["K6"])

    n12 = (n1, n2)
    m3456 = (m3, m4, m5, m6)
    k12 = (k1, k2)

    # ---------- LEFT swaps ----------
    K12_key = (K1 + K2) if (n1 == n2) else (K1, K2)
    K34_key = (K3 + K4) if (m3 == m4) else (K3, K4)
    K56_key = (K5 + K6) if (m5 == m6) else (K5, K6)

    # ---------- RIGHT: 1'<->2' ----------
    N12 = [N1, N2]
    M12 = [M1, M2]
    N_M_12_sq = sum(1 for Ni, Mi in zip(N12, M12) if Ni == 1 and Mi == 1)
    invars_12 = (sum(N12), sum(M12), N_M_12_sq)
    key12 = invars_12 if (k1 == k2) else (N1, N2, M1, M2)

    # ---------- RIGHT: 3'..6' via counts ----------
    N3456 = [N3, N4, N5, N6]
    M3456 = [M3, M4, M5, M6]
    N3456_sq = sum(N3456)
    M3456_sq = sum(M3456)
    N_M_3456_sq = sum(1 for Ni, Mi in zip(N3456, M3456) if Ni == 1 and Mi == 1)
    invariants_3456 = (N3456_sq, M3456_sq, N_M_3456_sq)

    return (n12, m3456, k12, K12_key, K34_key, K56_key, key12, invariants_3456)


def get_unique_representatives(solutions):
    if not solutions:
        return []
    key_to_idx = {}
    for i, sol in enumerate(solutions):
        key = equivalence_key(sol)
        if key not in key_to_idx:
            key_to_idx[key] = i
    return [solutions[i] for i in sorted(key_to_idx.values())]


def pack_solution(sol):
    return {
        "n12": (sol["n1"], sol["n2"]),
        "N": tuple(sol[f"N{i}"] for i in range(1, 7)),
        "m3456": (sol["m3"], sol["m4"], sol["m5"], sol["m6"]),
        "M": tuple(sol[f"M{i}"] for i in range(1, 7)),
        "k12": (sol["k1"], sol["k2"]),
        "K": tuple(sol[f"K{i}"] for i in range(1, 7)),
    }


# ------------- CLASSIFICATION HELPERS (I..VI and i..iii) ------------- #

def nm_class_label(n12, m3456):
    n1, n2 = n12
    m3, m4, m5, m6 = m3456

    if (n1, n2, m3, m4, m5, m6) == (0, 0, 0, 0, 0, 0):
        return "I"
    if (n1, n2, m3, m4, m5, m6) == (0, 0, 1, 0, 1, 0):
        return "II"
    if (n1, n2, m3, m4, m5, m6) == (0, 0, 1, 1, 1, 1):
        return "III"
    if (n1, n2, m3, m4, m5, m6) == (1, 0, 1, 0, 1, 1):
        return "IV"
    if (n1, n2, m3, m4, m5, m6) == (1, 1, 1, 1, 1, 1):
        return "V"
    if (n1, n2, m3, m4, m5, m6) == (1, 1, 1, 1, 0, 0):
        return "VI"

    return "UNKNOWN"


def k_class_label(k12):
    if k12 == (0, 0):
        return "i"
    if k12 == (1, 0):
        return "ii"
    if k12 == (1, 1):
        return "iii"
    return "UNKNOWN"


# ------------- MODULAR INVARIANCE CHECK FOR stefan SOLUTIONS ------------- #

def check_modular_invariance_B1B2B3(sol):
    n1, n2 = sol["n1"], sol["n2"]
    m3, m4, m5, m6 = sol["m3"], sol["m4"], sol["m5"], sol["m6"]
    k1, k2 = sol["k1"], sol["k2"]

    N = [sol[f"N{i}"] for i in range(1, 7)]
    M = [sol[f"M{i}"] for i in range(1, 7)]
    K = [sol[f"K{i}"] for i in range(1, 7)]

    n12_sq = n1 + n2
    m34_sq = m3 + m4
    m56_sq = m5 + m6
    N_sq = sum(N)
    M_sq = sum(M)
    k12_sq = k1 + k2
    K_sq = sum(K)

    NM_sq = sum((Ni - Mi) ** 2 for Ni, Mi in zip(N, M))

    c1 = ((n12_sq - N_sq) % 4 == 0)
    c2 = ((m34_sq - M_sq) % 4 == 0)
    c3 = ((m56_sq - NM_sq) % 4 == 0)
    c4 = ((k12_sq - K_sq) % 4 == 0)

    two_n12_minus_1 = [2 * n1 - 1, 2 * n2 - 1]
    two_k12_minus_1 = [2 * k1 - 1, 2 * k2 - 1]
    two_m34_minus_1 = [2 * m3 - 1, 2 * m4 - 1]

    N12 = N[:2]
    M12 = M[:2]
    K12v = K[:2]
    K34v = K[2:4]

    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    lhs_B1barB1 = dot(two_n12_minus_1, K12v) - dot(two_k12_minus_1, N12)
    rhs_B1barB1 = N_sq - K_sq
    c5 = ((lhs_B1barB1 - rhs_B1barB1) % 4 == 0)

    lhs_B2barB1 = dot(two_m34_minus_1, K34v) - dot(two_k12_minus_1, M12)
    rhs_B2barB1 = M_sq - K_sq
    c6 = ((lhs_B2barB1 - rhs_B2barB1) % 4 == 0)

    # E·B1·B2·B3 parity
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

    left_intersections = [E_left[i] * B1_left[i] * B2_left[i] * B3_left[i] for i in range(len(E_left))]
    right_intersections = [E_right[i] * B1_right[i] * B2_right[i] * B3_right[i] for i in range(len(E_right))]

    c7 = ((sum(left_intersections) - sum(right_intersections)) % 2 == 0)

    return c1 and c2 and c3 and c4 and c5 and c6 and c7


# ------------- LOAD stefan SOLUTIONS ------------- #

def _parse_tuple_cell(x, expected_len, colname):
    """
    Parse a cell like "(0, 1, 0)" into a tuple of ints.
    Works if x is already a tuple/list too.
    """
    if isinstance(x, (tuple, list)):
        t = tuple(int(v) for v in x)
    else:
        if pd.isna(x):
            raise ValueError(f"Missing value in column '{colname}'.")
        s = str(x).strip()
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


def load_stefan_solutions_from_csv(filename="Z2L_2_Z2R-reduced_stefan_formatted.csv"):
    """
    Supports two formats:

    A) Packed (your current file):
       columns: n12, m3456, k12, N, M, K
       with tuple strings like "(0, 0)" or "(0,0,0,0,0,0)".

    B) Expanded:
       columns: n1,n2,m3,m4,m5,m6,k1,k2,N1..N6,M1..M6,K1..K6
    """
    df = pd.read_csv(filename)

    cols = set(df.columns)

    # --- Packed format ---
    if {"n12", "m3456", "k12", "N", "M", "K"}.issubset(cols):
        sols = []
        for _, row in df.iterrows():
            n12 = _parse_tuple_cell(row["n12"], 2, "n12")
            m3456 = _parse_tuple_cell(row["m3456"], 4, "m3456")
            k12 = _parse_tuple_cell(row["k12"], 2, "k12")
            N = _parse_tuple_cell(row["N"], 6, "N")
            M = _parse_tuple_cell(row["M"], 6, "M")
            K = _parse_tuple_cell(row["K"], 6, "K")

            sol = {
                "n1": n12[0], "n2": n12[1],
                "m3": m3456[0], "m4": m3456[1], "m5": m3456[2], "m6": m3456[3],
                "k1": k12[0], "k2": k12[1],
            }
            sol.update({f"N{i}": N[i - 1] for i in range(1, 7)})
            sol.update({f"M{i}": M[i - 1] for i in range(1, 7)})
            sol.update({f"K{i}": K[i - 1] for i in range(1, 7)})

            sols.append(sol)
        return sols

    # --- Expanded format ---
    required = (
        ["n1", "n2", "m3", "m4", "m5", "m6", "k1", "k2"]
        + [f"N{i}" for i in range(1, 7)]
        + [f"M{i}" for i in range(1, 7)]
        + [f"K{i}" for i in range(1, 7)]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Unrecognized stefan CSV format. Missing columns:\n  "
            + ", ".join(missing)
            + "\nExpected either packed columns (n12,m3456,k12,N,M,K) or expanded columns."
        )

    sols = []
    for _, row in df.iterrows():
        sol = {c: int(row[c]) for c in required}
        sols.append(sol)
    return sols


# ------------- MAIN ------------- #

if __name__ == "__main__":
    # 1. Enumerate all Z3 models (B1,B2,B3)
    print("Enumerating all Z3 solutions for B1,B2,B3 with canonical (n12,m3456) classes...")
    all_solutions = enumerate_solutions()
    print(f"Total raw solutions found: {len(all_solutions)}")

    # 2. Reduce to unique representatives under the equivalence
    print("Computing unique representatives under equivalences (serial)...")
    unique_solutions = get_unique_representatives(all_solutions)
    print(f"Unique solutions after quotienting: {len(unique_solutions)}")

    # 3. Save unique solutions (RENAMED)
    rows = [pack_solution(sol) for sol in unique_solutions]
    df_unique = pd.DataFrame(rows, columns=["n12", "N", "m3456", "M", "k12", "K"])
    out_name_unique = "Z2L_2_Z2R_unique_code_solutions.csv"
    df_unique.to_csv(out_name_unique, index=False)
    print(f"Saved unique solutions to {out_name_unique}")

    # 3b. Count solutions by (k12 class) x (n12,m3456 class), compare to expected
    df_unique["nm_class"] = df_unique.apply(lambda r: nm_class_label(r["n12"], r["m3456"]), axis=1)
    df_unique["k_class"] = df_unique["k12"].apply(k_class_label)

    counts = pd.crosstab(df_unique["k_class"], df_unique["nm_class"]).reindex(
        index=["i", "ii", "iii"], columns=["I", "II", "III", "IV", "V", "VI"], fill_value=0
    )

    print("\nCounts from code (unique representatives):")
    print(counts)

    expected = pd.DataFrame(
        {
            "I":   {"i": 1, "ii": 0, "iii": 0},
            "II":  {"i": 1, "ii": 1, "iii": 1},
            "III": {"i": 1, "ii": 0, "iii": 1},
            "IV":  {"i": 1, "ii": 2, "iii": 6},
            "V":   {"i": 0, "ii": 2, "iii": 2},
            "VI":  {"i": 0, "ii": 0, "iii": 2},
        }
    ).reindex(index=["i", "ii", "iii"], columns=["I", "II", "III", "IV", "V", "VI"], fill_value=0)

    print("\nExpected counts (your table):")
    print(expected)

    diff = counts - expected
    print("\nDifference (code - expected):")
    print(diff)

    # 4. Load stefan solutions (packed tuples supported)
    stefan_file = "Z2L_2_Z2R-reduced_stefan_formatted.csv"
    print(f"\nLoading stefan solutions CSV ({stefan_file})...")
    stefan_solutions = load_stefan_solutions_from_csv(stefan_file)
    print(f"Loaded {len(stefan_solutions)} stefan solutions")

    # 5. Build map: equivalence_key -> index in unique_solutions (1-based)
    key_to_code_idx = {equivalence_key(sol): i + 1 for i, sol in enumerate(unique_solutions)}

    # 6. For each stefan solution, check modular invariance and equivalence class
    mapped_rows = []
    for stefan_idx_1based, sol in enumerate(stefan_solutions, start=1):
        n12 = (sol["n1"], sol["n2"])
        m3456 = (sol["m3"], sol["m4"], sol["m5"], sol["m6"])
        k12 = (sol["k1"], sol["k2"])
        N_vec = tuple(sol[f"N{i}"] for i in range(1, 7))
        M_vec = tuple(sol[f"M{i}"] for i in range(1, 7))
        K_vec = tuple(sol[f"K{i}"] for i in range(1, 7))

        row = {
            "stefan_index": stefan_idx_1based,
            "n12": n12,
            "m3456": m3456,
            "k12": k12,
            "N": N_vec,
            "M": M_vec,
            "K": K_vec,
        }

        row["modular_invariant"] = check_modular_invariance_B1B2B3(sol)
        key = equivalence_key(sol)
        row["equivalent_to_solution"] = key_to_code_idx.get(key, -1)  # -1 = no match
        row["nm_class"] = nm_class_label(n12, m3456)
        row["k_class"] = k_class_label(k12)

        mapped_rows.append(row)

    df_mapped = pd.DataFrame(mapped_rows)
    out_name_mapped = "stefan_reduced_mapped.csv"
    df_mapped.to_csv(out_name_mapped, index=False)
    print(f"\nSaved stefan solution mapping to {out_name_mapped}")

    # 7. Mapping diagnostics
    mapped_idxs = df_mapped["equivalent_to_solution"].tolist()
    matched = sorted([x for x in mapped_idxs if isinstance(x, int) and x > 0])

    all_unique = set(range(1, len(unique_solutions) + 1))
    missing = sorted(all_unique - set(matched))

    vc = pd.Series(matched).value_counts()
    dups = vc[vc > 1].sort_index()

    print("\n--- Mapping diagnostics (stefan -> code unique) ---")
    print(f"Unique solutions total: {len(unique_solutions)}")
    print(f"Stefan solutions total: {len(df_mapped)}")
    print(f"Matched (non -1) count: {len(matched)}")
    print(f"Unmatched (-1) count: {(df_mapped['equivalent_to_solution'] == -1).sum()}")

    print("\nSorted matched indices:")
    print(matched)

    print("\nMissing unique indices (no stefan solution mapped to these):")
    print(missing)

    print("\nDuplicate hits (unique_index: multiplicity):")
    print(dups.to_dict())

    collisions = (
        df_mapped[df_mapped["equivalent_to_solution"] > 0]
        .groupby("equivalent_to_solution")
        .apply(lambda g: list(g.index))  # stefan row indices (0-based)
    )
    collisions = collisions[collisions.apply(len) > 1].sort_index()

    print("\nCollisions (unique_index -> list of stefan row indices that map to it):")
    print(collisions.to_dict())

    pd.Series(matched, name="matched_indices_sorted").to_csv(
        "B1B2B3_mapping_matched_indices_sorted.csv", index=False
    )
    pd.Series(missing, name="missing_unique_indices").to_csv(
        "B1B2B3_mapping_missing_unique_indices.csv", index=False
    )
    dups.rename("multiplicity").to_csv("B1B2B3_mapping_duplicates.csv")
    collisions.rename("stefan_row_indices").to_csv("B1B2B3_mapping_collisions.csv")

    # 8. Reverse mapping output: code unique solutions + stefan index (or UNMATCHED)
    #    If multiple stefan solutions hit the same code solution, we store "a,b,c".
    code_to_stefan = (
        df_mapped[df_mapped["equivalent_to_solution"] > 0]
        .groupby("equivalent_to_solution")["stefan_index"]
        .apply(lambda s: ",".join(str(int(x)) for x in sorted(s.tolist())))
        .to_dict()
    )

    df_unique_rev = df_unique.copy()
    df_unique_rev.insert(
        len(df_unique_rev.columns),
        "mapped_stefan_index",
        [code_to_stefan.get(i + 1, "UNMATCHED") for i in range(len(df_unique_rev))]
    )

    out_name_reverse = "Z2L_2_Z2R_unique_code_solutions_mapped.csv"
    df_unique_rev.to_csv(out_name_reverse, index=False)
    print(f"\nSaved reverse mapping (code -> stefan) to {out_name_reverse}")
