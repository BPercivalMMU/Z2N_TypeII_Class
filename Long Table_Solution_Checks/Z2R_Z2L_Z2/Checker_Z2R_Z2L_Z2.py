# -*- coding: utf-8 -*-
"""
Z2L x Z2R x Z2 basis vector classification with B1, B2, Z2.

Equivalence key uses:
  (n12, m3456, N12, N34, N56, K12, K34, K56)
"""

from z3 import *
import pandas as pd
from collections import Counter

# ------------- BUILD SOLVER ------------- #

def build_solver():
    # define input variables for Z2L x Z2R x Z2
    n1, n2 = Ints('n1 n2')
    N1, N2, N3, N4, N5, N6 = Ints('N1 N2 N3 N4 N5 N6')
    k1, k2 = Ints('k1 k2')
    K1, K2, K3, K4, K5, K6 = Ints('K1 K2 K3 K4 K5 K6')
    m3, m4, m5, m6 = Ints('m3 m4 m5 m6')
    mb3, mb4, mb5, mb6 = Ints('mb3 mb4 mb5 mb6')
    
    Z2_BCs = [
        n1, n2,
        N1, N2, N3, N4, N5, N6,
        k1, k2,
        K1, K2, K3, K4, K5, K6,
        m3, m4, m5, m6,
        mb3, mb4, mb5, mb6
    ]

    s = Solver()

    # All parameters are bits: 0 or 1
    for x in Z2_BCs:
        s.add(Or(x == 0, x == 1))
    
    # symmetric Z2 
    s.add(And(m3 == mb3, m4 == mb4, m5 == mb5, m6 == mb6))
        
    # n12, k12, m56 ∈ {(00),(10),(11)}, m34 ∈ {(00),(10)}
    s.add(Or(And(n1 == 0, n2 == 0),
             And(n1 == 1, n2 == 0),
             And(n1 == 1, n2 == 1)))
    s.add(Or(And(k1 == 0, k2 == 0),
             And(k1 == 1, k2 == 0),
             And(k1 == 1, k2 == 1)))
    s.add(Or(And(m3 == 0, m4 == 0),
             And(m3 == 1, m4 == 0)))
    s.add(Or(And(m5 == 0, m6 == 0),
             And(m5 == 1, m6 == 0),
             And(m5 == 1, m6 == 1)))
    
    # at most 3 entries in N and K  non-zero
    N_vec  = [N1, N2, N3, N4, N5, N6]
    NN     = Sum(N_vec)
    K_vec  = [K1, K2, K3, K4, K5, K6]
    KK     = Sum(K_vec)
    s.add(NN <= 3)
    s.add(KK <= 3)

    ##-----Modular invariance conditions-------------------------
    
    # aux variables
    n12_sq  = n1 + n2
    k12_sq  = k1 + k2
    m34_sq  = m3 + m4

    # MI conditions 5.4 a)
    B1B1 = [n12_sq == NN]
    B2B2 = [k12_sq == KK]
    
    # MI conditions 5.4 c) for B1 and Z2
    B1Z2_dot = (2 + n12_sq + m34_sq + (1 - m5) + (1 - m6)) - (
                N1 + N2 + 2*(m3*N3 + m4*N4) + N5 + N6)
    B1Z2 = [B1Z2_dot % 4 == 0]
    
    # MI conditions 5.4 c) for B2 and Z2
    B2Z2_dot = (K1 + K2 + 2*(m3*K3 + m4*K4) + K5 + K6)- (
                2 + k12_sq + m34_sq + (1 - m5) + (1 - m6))
    B2Z2 = [B2Z2_dot % 4 == 0]
    
    # B1 . B2 modular invariance
    lhs_B1B2 = 2*n1*K1 + 2*n2*K2 + K3 + K4 + K5 + K6
    rhs_B1B2 = 2*k1*N1 + 2*k2*N2 + N3 + N4 + N5 + N6
    B1B2 = [(lhs_B1B2 - rhs_B1B2) % 4 == 0]
    
    # no real fermions
    EB1B2Z2_lhs = n1*K1+n2*K2+K3*m3+K4*m4+K5*(1-m5)+K6*(1-m6)
    EB1B2Z2_rhs = k1*N1+k2*N2+N3*m3+N4*m4+N5*(1-m5)+N6*(1-m6)
    EB1B2Z2 = [(EB1B2Z2_lhs-EB1B2Z2_rhs)%2 == 0]
    
    MI_constraints = B1B1 + B2B2 + B1Z2 + B2Z2 + B1B2 + EB1B2Z2
    s.add(MI_constraints) 

    # ----- Table 3 classes for m3456 -----
    # (0^4),(0^2 10),(10 0^2),(0^2 1^2),(10 10)
    A = And(m3 == 0, m4 == 0, m5 == 0, m6 == 0)
    B = And(m3 == 0, m4 == 0, m5 == 1, m6 == 0)
    C = And(m3 == 1, m4 == 0, m5 == 0, m6 == 0)
    D = And(m3 == 0, m4 == 0, m5 == 1, m6 == 1)
    E = And(m3 == 1, m4 == 0, m5 == 1, m6 == 0)

    s.add(Or(A, B, C, D, E))

    # RETURN solver and variable list
    return s, Z2_BCs


# ------------- ENUMERATION OF ALL SOLUTIONS ------------- #

def enumerate_solutions():
    s, vars_ = build_solver()
    sols = []

    while s.check() == sat:
        m = s.model()
        sol = {}
        for v in vars_:
            val = m.eval(v, model_completion=True)
            sol[v.decl().name()] = val.as_long()
        sols.append(sol)

        # Block this exact solution:
        s.add(Or(*[v != m.eval(v, model_completion=True) for v in vars_]))

    return sols


# ------------- EQUIVALENCE KEY ------------- #

def equivalence_key(sol):
    """
    Equivalence classes are labeled by:

      n12   = (n1, n2)
      m3456 = (m3, m4, m5, m6)
      N12   = N1 + N2
      N34   = N3+N4 if m3 == m4, else (N3, N4)
      N56   = N5+N6 if m5 == m6, else (N5, N6)
      K12   = K1 + K2
      K34   = K3+K4 if m3 == m4, else (K3, K4)
      K56   = K5+K6 if m5 == m6, else (K5, K6)
    """
    n1, n2 = sol["n1"], sol["n2"]
    k1, k2 = sol["k1"], sol["k2"]
    m3, m4, m5, m6 = sol["m3"], sol["m4"], sol["m5"], sol["m6"]
    N1, N2, N3, N4, N5, N6 = (sol["N1"], sol["N2"], sol["N3"],
                              sol["N4"], sol["N5"], sol["N6"])
    K1, K2, K3, K4, K5, K6 = (sol["K1"], sol["K2"], sol["K3"],
                              sol["K4"], sol["K5"], sol["K6"])

    n12    = (n1, n2)
    m3456  = (m3, m4, m5, m6)
    
    if n1 == n2:
        K12_key = K1+K2
    else:
        K12_key = (K1, K2)
    
    if k1 == k2:
        N12_key = N1+N2
    else:
        N12_key = (N1, N2)
        
    if m5 == m6:
        N56_key = N5 + N6
        K56_key = K5 + K6
    else:
        N56_key = (N5, N6)
        K56_key = (K5, K6)
        
    if m3 == m4:
        N34_key = N3 + N4
        K34_key = K3 + K4
    else:
        N34_key = (N3, N4)
        K34_key = (K3, K4)
        
    if m5 == m6:
        N56_key = N5 + N6
        K56_key = K5 + K6
    else:
        N56_key = (N5, N6)
        K56_key = (K5, K6)


    key = (n12, m3456, N12_key,  N34_key, N56_key, K12_key, K34_key, K56_key)
    return key


def pack_solution(sol):
    """Return row dict for CSV."""
    return {
        "n12":    (sol["n1"], sol["n2"]),
        "N":      tuple(sol[f"N{i}"] for i in range(1, 7)),
        "k12":    (sol["k1"], sol["k2"]),
        "K":      tuple(sol[f"K{i}"] for i in range(1, 7)),
        "m3456":  (sol["m3"], sol["m4"], sol["m5"], sol["m6"]),
        "mb3456": (sol["mb3"], sol["mb4"], sol["mb5"], sol["mb6"]),
    }


def get_unique_representatives(solutions):
    """
    Given a list of solution dicts, return a list of unique representatives
    under the equivalence defined by equivalence_key.
    """
    if not solutions:
        return []

    idx_key_list = [(i, equivalence_key(sol))
                    for i, sol in enumerate(solutions)]

    key_to_idx = {}
    for idx, key in idx_key_list:
        if key not in key_to_idx:
            key_to_idx[key] = idx

    unique_solutions = [solutions[i] for i in sorted(key_to_idx.values())]
    return unique_solutions


# ------------- CHECK stefan SOLUTIONS AGAINST CONSTRAINTS ------------- #

def check_constraints_for_solution(sol):
    """
    Given a dict sol with all variables, check whether it satisfies
    all constraints in build_solver().
    """
    s, vars_ = build_solver()
    # Pin all variables to the values in sol
    for v in vars_:
        name = v.decl().name()
        if name not in sol:
            raise KeyError(f"stefan solution missing variable {name}")
        s.add(v == sol[name])

    return s.check() == sat


# ------------- MAIN ------------- #

if __name__ == "__main__":
    # 1. Enumerate all Z3 models under the full constraints
    print("Enumerating all Z3 solutions (B1, B2, Z2)...")
    all_solutions = enumerate_solutions()
    print(f"Total raw solutions found: {len(all_solutions)}")

    # 2. Reduce to unique representatives under equivalence
    print("Computing unique representatives under equivalence...")
    unique_solutions = get_unique_representatives(all_solutions)
    print(f"Unique solutions after quotienting: {len(unique_solutions)}")

    # 3. Pack into rows and save
    rows = [pack_solution(sol) for sol in unique_solutions]
    df_unique = pd.DataFrame(
        rows,
        columns=["n12", "N", "k12", "K", "m3456", "mb3456"]
    )
    out_name = "B1_B2_Z2_unique_solutions.csv"
    df_unique.to_csv(out_name, index=False)
    print(f"Saved unique solutions to {out_name}")

    # 4. Read CSV of your stefan solutions and get them as dicts
    df_stefan = pd.read_csv("B1B2Z2_stefan_solutions.csv", header=0)

    expected_cols = [
        "n1", "n2",
        "N1", "N2", "N3", "N4", "N5", "N6",
        "K1", "K2", "K3", "K4", "K5", "K6",
        "k1", "k2",
        "m3", "m4", "m5", "m6"
    ]
    if len(df_stefan.columns) != len(expected_cols):
        raise ValueError(
            f"Expected {len(expected_cols)} columns, got {len(df_stefan.columns)}: "
            f"{list(df_stefan.columns)}"
        )
    df_stefan.columns = expected_cols

    stefan_solutions = []
    for _, row in df_stefan.iterrows():
        sol = {
            "n1":  int(row["n1"]),
            "n2":  int(row["n2"]),
            "N1":  int(row["N1"]),
            "N2":  int(row["N2"]),
            "N3":  int(row["N3"]),
            "N4":  int(row["N4"]),
            "N5":  int(row["N5"]),
            "N6":  int(row["N6"]),
            "K1":  int(row["K1"]),
            "K2":  int(row["K2"]),
            "K3":  int(row["K3"]),
            "K4":  int(row["K4"]),
            "K5":  int(row["K5"]),
            "K6":  int(row["K6"]),
            "k1":  int(row["k1"]),
            "k2":  int(row["k2"]),
            "m3":  int(row["m3"]),
            "m4":  int(row["m4"]),
            "m5":  int(row["m5"]),
            "m6":  int(row["m6"]),
        }
        sol["mb3"] = sol["m3"]
        sol["mb4"] = sol["m4"]
        sol["mb5"] = sol["m5"]
        sol["mb6"] = sol["m6"]

        stefan_solutions.append(sol)

    # 5. Build map from equivalence key -> index in unique_solutions (1-based)
    key_to_idx = {
        equivalence_key(sol): i + 1
        for i, sol in enumerate(unique_solutions)
    }

    # 6. For each stefan solution, check constraints and equivalence class
    mapped_rows = []
    for sol in stefan_solutions:
        n12    = (sol["n1"], sol["n2"])
        k12    = (sol["k1"], sol["k2"])
        N_vec  = tuple(sol[f"N{i}"] for i in range(1, 7))
        K_vec  = tuple(sol[f"K{i}"] for i in range(1, 7))
        m3456  = (sol["m3"], sol["m4"], sol["m5"], sol["m6"])
        mb3456 = (sol["mb3"], sol["mb4"], sol["mb5"], sol["mb6"])

        row = {
            "n12":    n12,
            "N":      N_vec,
            "k12":    k12,
            "K":      K_vec,
            "m3456":  m3456,
            "mb3456": mb3456,
        }

        ok = check_constraints_for_solution(sol)
        row["satisfies_all_constraints"] = ok

        key = equivalence_key(sol)
        row["equivalent_to_solution_index"] = key_to_idx.get(key, -1)

        mapped_rows.append(row)

    df_mapped = pd.DataFrame(mapped_rows)
    out_name_mapped = "B1B2Z2_stefan_mapped.csv"
    df_mapped.to_csv(out_name_mapped, index=False)
    print(f"Saved stefan solution mapping to {out_name_mapped}")

    # 7. Analyse mapped indices: sorted list, repeats, and missing values
    indices = df_mapped["equivalent_to_solution_index"].tolist()

    # Exclude -1 (no match)
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
        print("\nNo repeated indices (among matched ones).")

    # Missing indices between 1 and number of unique solutions
    expected = set(range(1, len(unique_solutions) + 1))
    missing = sorted(expected - set(sorted_indices))
    print(f"\nMissing indices in 1..{len(unique_solutions)} (i.e. "
          f"solutions not hit by any stefan example):")
    print(missing)

    # Optional: how many -1s?
    num_unmatched = len([i for i in indices if i == -1])
    if num_unmatched:
        print(f"\nNumber of stefan solutions with no matching unique solution (index = -1): "
              f"{num_unmatched}")
