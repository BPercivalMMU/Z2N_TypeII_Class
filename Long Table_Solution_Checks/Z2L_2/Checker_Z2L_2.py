# Z2R_2 solutions checker
# convention: avoided barred naming so B1b -> B2, (and will use: B2 -> B3, B2b -> B4)

from z3 import *
import pandas as pd
from multiprocessing import Pool, cpu_count
from collections import Counter


# ------------- Z3 MODEL BUILDING FOR B1, B2 ONLY ------------- #

def build_solver():
    # ----- Declare 0/1 integer variables -----
    # B1 parameters
    n1, n2 = Ints('n1 n2')
    N1, N2, N3, N4, N5, N6 = Ints('N1 N2 N3 N4 N5 N6')

    # B2 parameters
    m3, m4, m5, m6 = Ints('m3 m4 m5 m6')
    M1, M2, M3, M4, M5, M6 = Ints('M1 M2 M3 M4 M5 M6')

    Z2_BCs = [n1, n2,
              N1, N2, N3, N4, N5, N6,
              m3, m4, m5, m6,
              M1, M2, M3, M4, M5, M6]

    s = Solver()

    # All variables are bits: 0 or 1
    for x in Z2_BCs:
        s.add(Or(x == 0, x == 1))

    # ----- restrictions A1 & A3 ----- #

    # A1: n12 ∈ {(00), (10), (11)}
    s.add(Or(And(n1 == 0, n2 == 0),
             And(n1 == 1, n2 == 0),
             And(n1 == 1, n2 == 1)))

    # A3: m34, m56 ∈ {(00), (10), (11)}
    s.add(Or(And(m3 == 0, m4 == 0),
             And(m3 == 1, m4 == 0),
             And(m3 == 1, m4 == 1)))  
    s.add(Or(And(m5 == 0, m6 == 0),
             And(m5 == 1, m6 == 0),
             And(m5 == 1, m6 == 1)))
    
    # At most 3 non-zero entries in N and M
    N = [N1, N2, N3, N4, N5, N6]
    M = [M1, M2, M3, M4, M5, M6]
    s.add(Sum(N) <= 3)
    s.add(Sum(M) <= 3)

    # ----- Modular invariance constraints for B1, B2 ----- #

    # "Squares" as sums of components
    NN  = Sum(N)
    MM  = Sum(M)

    # (N - M)^2 as Hamming distance square
    NM_sq = Sum((Ni - Mi) * (Ni - Mi) for Ni, Mi in zip(N, M))

    # 1) n_{12}^2 = N^2 (mod 4)
    s.add((n1 + n2 - NN) % 4 == 0)

    # 2) m_{34}^2 = M^2 (mod 4)
    s.add((m3 + m4 - MM) % 4 == 0)

    # 3) m_{56}^2 = (N - M)^2 (mod 4)
    s.add((m5 + m6 - NM_sq) % 4 == 0)

    return s, Z2_BCs


# ------------- ENUMERATION OF ALL SOLUTIONS ------------- #

def enumerate_solutions():
    s, vars_ = build_solver()
    sols = []

    while s.check() == sat:
        m = s.model()
        sol = {str(v): m[v].as_long() for v in vars_}
        sols.append(sol)

        # Block this exact model
        s.add(Or(*[v != m[v] for v in vars_]))

    return sols


# ------------- EQUIVALENCE KEY (B1,B2 ONLY, FIXED) ------------- #

def equivalence_key_B1B2(sol): 
    """ Equivalence key for B1,B2 using the squared lengths (|N|^2, |M|^2, |N-M|^2), 
    in line with Table 4 """ 
    N = [sol["N1"], sol["N2"], sol["N3"], sol["N4"], sol["N5"], sol["N6"]] 
    M = [sol["M1"], sol["M2"], sol["M3"], sol["M4"], sol["M5"], sol["M6"]] 
    N_sq = sum(N) 
    M_sq = sum(M) 
    NM_sq = sum((Ni - Mi) * (Ni - Mi) for Ni, Mi in zip(N, M)) 
    triple = [N_sq, M_sq, NM_sq] 
    triple.sort() # order doesnt matter
    return tuple(triple)



def pack_solution_B1B2(sol):
    """Pack solution into 4 columns for the CSV."""
    return {
        "n12":   (sol["n1"], sol["n2"]),
        "N":     tuple(sol[f"N{i}"] for i in range(1, 7)),
        "m3456": (sol["m3"], sol["m4"], sol["m5"], sol["m6"]),
        "M":     tuple(sol[f"M{i}"] for i in range(1, 7)),
    }


# ------------- PARALLEL POST-PROCESSING ------------- #

# =============================================================================
# def compute_keys_parallel_B1B2(solutions, n_procs=None):
#     if n_procs is None:
#         n_procs = max(1, cpu_count() - 1)
# 
#     if n_procs == 1 or len(solutions) == 0:
#         return [equivalence_key_B1B2(sol) for sol in solutions]
# 
#     with Pool(processes=n_procs) as pool:
#         keys = pool.map(equivalence_key_B1B2, solutions)
#     return keys
# =============================================================================


def get_unique_representatives_B1B2(solutions, parallel=True):
    keys = [equivalence_key_B1B2(sol) for sol in solutions]

    seen = {}
    for sol, key in zip(solutions, keys):
        if key not in seen:
            seen[key] = sol

    return list(seen.values())


# ------------- CHECK HAND SOLUTIONS AGAINST CONSTRAINTS ------------- #

def check_constraints_for_solution(sol):
    """
    Given a dict sol with all variables, check whether it satisfies
    all constraints in build_solver().
    """
    s, vars_ = build_solver()
    for v in vars_:
        name = str(v)
        if name not in sol:
            raise KeyError(f"Hand solution missing variable {name}")
        s.add(v == sol[name])

    return s.check() == sat


# ------------- MAIN ------------- #

if __name__ == "__main__":

    # 1. Enumerate all Z3 models for B1,B2
    print("Enumerating all Z3 solutions for B1,B2...")
    all_solutions = enumerate_solutions()
    print(f"Total raw solutions found (B1,B2): {len(all_solutions)}")

    # 2. Reduce to unique representatives under the improved equivalence
    print("Computing unique representatives (B1,B2)...")
    unique_solutions = get_unique_representatives_B1B2(all_solutions, parallel=True)
    print(f"Unique solutions after quotienting (B1,B2): {len(unique_solutions)}")

    # 2a. Sort unique_solutions by (n1, n2) so order is (0,0), (1,0), (1,1)
    unique_solutions.sort(key=lambda s: (s["n1"], s["n2"]))

    # 3. Pack into rows and save (already sorted by n1,n2)
    rows = [pack_solution_B1B2(sol) for sol in unique_solutions]

    df_unique = pd.DataFrame(rows, columns=["n12", "N", "m3456", "M"])
    out_name = "Z2L_2_unique_solutions.csv"
    df_unique.to_csv(out_name, index=False)

    print(f"Saved unique solutions (B1,B2) to {out_name}")

    # 4. Read Stefan's hand solutions and check / map them
    # Columns:
    # n1 n2 N1 N2 N3 N4 N5 N6 m3 m4 m5 m6 M1 M2 M3 M4 M5 M6
    df_hand = pd.read_csv("Z2L_2_stefan_solutions.csv", header=0)

    expected_cols = [
        "n1", "n2",
        "N1", "N2", "N3", "N4", "N5", "N6",
        "m3", "m4", "m5", "m6",
        "M1", "M2", "M3", "M4", "M5", "M6",
    ]
    if len(df_hand.columns) != len(expected_cols):
        raise ValueError(
            f"Expected {len(expected_cols)} columns, got {len(df_hand.columns)}: "
            f"{list(df_hand.columns)}"
        )
    df_hand.columns = expected_cols  # force exact names

    hand_solutions = []
    for _, row in df_hand.iterrows():
        sol = {
            "n1": int(row["n1"]),
            "n2": int(row["n2"]),
            "N1": int(row["N1"]),
            "N2": int(row["N2"]),
            "N3": int(row["N3"]),
            "N4": int(row["N4"]),
            "N5": int(row["N5"]),
            "N6": int(row["N6"]),
            "m3": int(row["m3"]),
            "m4": int(row["m4"]),
            "m5": int(row["m5"]),
            "m6": int(row["m6"]),
            "M1": int(row["M1"]),
            "M2": int(row["M2"]),
            "M3": int(row["M3"]),
            "M4": int(row["M4"]),
            "M5": int(row["M5"]),
            "M6": int(row["M6"]),
        }
        hand_solutions.append(sol)

    # 5. Build map from equivalence key -> index in unique_solutions (1-based)
    key_to_idx = {
        equivalence_key_B1B2(sol): i + 1
        for i, sol in enumerate(unique_solutions)
    }

    # 6. For each hand solution, check constraints and equivalence class
    mapped_rows = []
    for sol in hand_solutions:
        n12    = (sol["n1"], sol["n2"])
        N_vec  = tuple(sol[f"N{i}"] for i in range(1, 7))
        m3456  = (sol["m3"], sol["m4"], sol["m5"], sol["m6"])
        M_vec  = tuple(sol[f"M{i}"] for i in range(1, 7))

        row = {
            "n12":   n12,
            "N":     N_vec,
            "m3456": m3456,
            "M":     M_vec,
        }

        ok = check_constraints_for_solution(sol)
        row["satisfies_all_constraints"] = ok

        key = equivalence_key_B1B2(sol)
        # -1 = no match in found unique solutions
        row["equivalent_to_solution_index"] = key_to_idx.get(key, -1)

        mapped_rows.append(row)

    df_mapped = pd.DataFrame(mapped_rows)
    out_name_mapped = "Z2L_2_stefan_mapped.csv"
    df_mapped.to_csv(out_name_mapped, index=False)
    print(f"Saved hand solution mapping to {out_name_mapped}")

    # 7. Analyse mapped indices: print raw, sorted, repeats, and missing values
    indices = df_mapped["equivalent_to_solution_index"].tolist()
    print("\nMapped indices from Stefan's file (including -1 for unmatched):")
    print(indices)

    valid_indices = [i for i in indices if i != -1]
    sorted_indices = sorted(valid_indices)
    print("\nSorted mapped indices (excluding -1):")
    print(sorted_indices)

    counts = Counter(sorted_indices)
    repeated = {idx: cnt for idx, cnt in counts.items() if cnt > 1}
    if repeated:
        print("\nRepeated indices (value: count):")
        for idx, cnt in sorted(repeated.items()):
            print(f"  {idx}: {cnt} times")
    else:
        print("\nNo repeated indices among matched hand solutions.")

    # Missing indices between 1 and number of unique solutions
    expected = set(range(1, len(unique_solutions) + 1))
    missing = sorted(expected - set(sorted_indices))
    print(f"\nMissing indices in 1..{len(unique_solutions)} "
          f"(unique solutions not hit by any hand example):")
    print(missing)

    # Count unmatched hand solutions
    num_unmatched = len([i for i in indices if i == -1])
    if num_unmatched:
        print(f"\nNumber of Stefan hand solutions with no matching unique solution (index = -1): "
              f"{num_unmatched}")
    else:
        print("\nAll Stefan hand solutions mapped to some unique solution.")
