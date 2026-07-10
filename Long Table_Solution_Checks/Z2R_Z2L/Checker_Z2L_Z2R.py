# Z2R_Z2L solution checker

from z3 import *
import pandas as pd

# ------------------------- Z3 MODEL (B1, B3) ------------------------- #

def build_solver():
    # B1 parameters
    n1, n2 = Ints('n1 n2')  # left-moving twist bits in directions 1,2
    N1, N2, N3, N4, N5, N6 = Ints('N1 N2 N3 N4 N5 N6')  # right-moving B1 twist

    # B3 parameters
    k1, k2 = Ints('k1 k2')  # right-moving B3 twist in 1',2'
    K1, K2, K3, K4, K5, K6 = Ints('K1 K2 K3 K4 K5 K6')  # left-moving B3 twist

    all_bits = [n1, n2, N1, N2, N3, N4, N5, N6,
                k1, k2, K1, K2, K3, K4, K5, K6]

    s = Solver()

    # All variables 0 or 1
    for b in all_bits:
        s.add(Or(b == 0, b == 1))

    # n12, k12: (00), (10), (11)
    
    s.add(Or(And(n1 == 0, n2 == 0),
             And(n1 == 1, n2 == 0),
             And(n1 == 1, n2 == 1)))
    
    s.add(Or(And(k1 == 0, k2 == 0),
             And(k1 == 1, k2 == 0),
             And(k1 == 1, k2 == 1)))
    
    # at most 3 non-zero entries in N and K
    N = [N1, N2, N3, N4, N5, N6]  # right-moving B1 twist
    K = [K1, K2, K3, K4, K5, K6]  # left-moving  B3 twist
    s.add(Sum(N) <= 3)
    s.add(Sum(K) <= 3)

    # Modular invariance constraints for B1, B3
    n12_sq  = n1 + n2
    N_sq    = Sum(N)
    k12_sq  = k1 + k2
    K_sq    = Sum(K)

    # (1) n_{12}^2 ≡ N^2 (mod 4)
    s.add((n12_sq - N_sq) % 4 == 0)

    # (2) k_{12}^2 ≡ K^2 (mod 4)  (bar n_12^2 = bar N^2)
    s.add((k12_sq - K_sq) % 4 == 0)

    # (3) B1 . B3
    lhs = 2*(n1*K1+n2*K2)+K3+K4+K5+K6
    rhs = 2*(k1*N1+k2*N2)+N3+N4+N5+N6
    s.add((lhs - rhs) % 4 == 0)

    return s, all_bits


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


# ------------------------- EQUIVALENCE: 1–2 and 3–4–5–6 ------------------------- #

def canonical_key_B1B3_uncoupled_3456(sol):
    """
    Equivalence for B1,B3:

    - Keep n12 and k12 as twist labels in (1,2) and (1',2').

    - LEFT 1,2:
        If n12 = (0,0) or (1,1), we may swap directions 1<->2.
        This acts on (K1,K2): so (K1,K2) ~ (K2,K1).
        If n12 = (1,0), no swap: (K1,K2) is fixed.

    - RIGHT 1',2':
        If k12 = (0,0) or (1,1), we may swap directions 1'<->2'.
        This acts on (N1,N2): so (N1,N2) ~ (N2,N1).
        If k12 = (1,0), no swap: (N1,N2) is fixed.

    - LEFT 3..6:
        We can permute 3,4,5,6 arbitrarily: only multiset {K3,K4,K5,K6} matters.

    - RIGHT 3'..6':
        We can permute 3',4',5',6' arbitrarily: only multiset {N3,N4,N5,N6} matters.

    No coupling between N_i and K_i at the same index.
    """

    n12 = (sol["n1"], sol["n2"])
    k12 = (sol["k1"], sol["k2"])

    N12_raw = (sol["N1"], sol["N2"])
    K12_raw = (sol["K1"], sol["K2"])

    # LEFT:  swap of K1,K2 if n12 symmetric
    if n12 in ((0,0), (1,1)):
        K12_candidates = [K12_raw, (K12_raw[1], K12_raw[0])]
        K12 = min(K12_candidates)
    else:
        K12 = K12_raw

    # RIGHT:  swap of N1,N2 if k12 symmetric
    if k12 in ((0,0), (1,1)):
        N12_candidates = [N12_raw, (N12_raw[1], N12_raw[0])]
        N12 = min(N12_candidates)
    else:
        N12 = N12_raw

    # Internal directions 3..6: independent permutations left/right
    N_rest = [sol[f"N{i}"] for i in range(3, 7)]  # N3..N6
    K_rest = [sol[f"K{i}"] for i in range(3, 7)]  # K3..K6

    N_rest_sorted = tuple(sorted(N_rest))
    K_rest_sorted = tuple(sorted(K_rest))

    return (n12, k12, N12, K12, N_rest_sorted, K_rest_sorted)


def dedupe_by_key(solutions, key_func):
    seen = {}
    for sol in solutions:
        k = key_func(sol)
        if k not in seen:
            seen[k] = sol
    return list(seen.values())


# ------------------------- INVARIANTS PACKING ------------------------- #

def pack_with_invariants(sol):
    def vec(prefix, length):
        return [sol[f"{prefix}{i}"] for i in range(1, length+1)]

    N = vec("N", 6)
    K = vec("K", 6)
    N12, Nrest = N[:2], N[2:]
    K12, Krest = K[:2], K[2:]

    def norm_sq(v): return sum(v)
    def dist_sq(a,b): return sum((ai-bi)**2 for ai,bi in zip(a,b))

    N_sq = norm_sq(N)
    K_sq = norm_sq(K)
    NK_sq = dist_sq(N, K)

    N12_sq = norm_sq(N12)
    K12_sq = norm_sq(K12)
    N12mK12_sq = dist_sq(N12, K12)

    Nrest_sq = norm_sq(Nrest)
    Krest_sq = norm_sq(Krest)
    NrestmKrest_sq = dist_sq(Nrest, Krest)

    return {
        "n12": (sol["n1"], sol["n2"]),
        "k12": (sol["k1"], sol["k2"]),
        "N": tuple(N),
        "K": tuple(K),
        "N_sq": N_sq,
        "K_sq": K_sq,
        "N_minus_K_sq": NK_sq,
        "N12_sq": N12_sq,
        "K12_sq": K12_sq,
        "N12_minus_K12_sq": N12mK12_sq,
        "Nrest_sq": Nrest_sq,
        "Krest_sq": Krest_sq,
        "Nrest_minus_Krest_sq": NrestmKrest_sq,
    }


# ------------------------- MODULAR CHECK ------------------------- #

def check_modular_invariance(sol):
    n1, n2 = sol["n1"], sol["n2"]
    k1, k2 = sol["k1"], sol["k2"]
    N = [sol[f"N{i}"] for i in range(1,7)]
    K = [sol[f"K{i}"] for i in range(1,7)]

    n12_sq = n1 + n2
    bn12_sq = k1 + k2
    N_sq = sum(N)
    K_sq = sum(K)

    c1 = ((n12_sq - N_sq) % 4 == 0)
    c2 = ((bn12_sq - K_sq) % 4 == 0)

    N12, K12 = N[:2], K[:2]
    two_n12_minus_1 = [2*n1 - 1, 2*n2 - 1]
    two_k12_minus_1 = [2*k1 - 1, 2*k2 - 1]

    lhs = sum(a*b for a,b in zip(two_n12_minus_1, K12)) \
        - sum(a*b for a,b in zip(two_k12_minus_1, N12))
    rhs = N_sq - K_sq
    c3 = ((lhs - rhs) % 4 == 0)
    
    return c1 and c2 and c3


# ------------------------- MAIN ------------------------- #

if __name__ == "__main__":
    print("Enumerating all Z3 solutions for B1,B3...")
    all_solutions = enumerate_solutions()
    print(f"Total raw solutions found (B1,B3): {len(all_solutions)}")

    # Deduplicate under this refined equivalence
    unique_solutions = dedupe_by_key(all_solutions, canonical_key_B1B3_uncoupled_3456)
    print(f"# unique under refined 1–2 & 3–4–5–6 equivalence: {len(unique_solutions)}")

    # Save unique solutions with invariants
    df_unique = pd.DataFrame([pack_with_invariants(s) for s in unique_solutions])
    df_unique.to_csv("B1B3_unique_solutions.csv", index=False)
    print("Saved unique classes to B1B3_unique_solutions.csv")

    # ----- Read by-stefan solutions from CSV instead of hard-coding -----
    # Expected column order: n1 n2 N1..N6 k1 k2 K1..K6
    df_stefan_in = pd.read_csv("B1B3_stefan_solutions.csv", header=0)

    expected_cols = [
        "n1", "n2",
        "N1", "N2", "N3", "N4", "N5", "N6",
        "k1", "k2",
        "K1", "K2", "K3", "K4", "K5", "K6",
    ]
    if len(df_stefan_in.columns) != len(expected_cols):
        raise ValueError(
            f"Expected {len(expected_cols)} columns, got {len(df_stefan_in.columns)}: "
            f"{list(df_stefan_in.columns)}"
        )
    df_stefan_in.columns = expected_cols  # enforce exact names

    stefan_solutions = []
    for _, row in df_stefan_in.iterrows():
        sol = {
            "n1": int(row["n1"]),
            "n2": int(row["n2"]),
            "k1": int(row["k1"]),
            "k2": int(row["k2"]),
            "N1": int(row["N1"]),
            "N2": int(row["N2"]),
            "N3": int(row["N3"]),
            "N4": int(row["N4"]),
            "N5": int(row["N5"]),
            "N6": int(row["N6"]),
            "K1": int(row["K1"]),
            "K2": int(row["K2"]),
            "K3": int(row["K3"]),
            "K4": int(row["K4"]),
            "K5": int(row["K5"]),
            "K6": int(row["K6"]),
        }
        stefan_solutions.append(sol)

    # Map stefan solutions
    key_to_idx = {
        canonical_key_B1B3_uncoupled_3456(s): i+1
        for i,s in enumerate(unique_solutions)
    }

    stefan_rows = []
    for sol in stefan_solutions:
        row = pack_with_invariants(sol)
        row["modular_invariant"] = check_modular_invariance(sol)
        row["equiv_refined_3456"] = key_to_idx.get(
            canonical_key_B1B3_uncoupled_3456(sol), -1
        )
        stefan_rows.append(row)

    df_stefan = pd.DataFrame(stefan_rows)
    df_stefan.to_csv("B1B3_stefan_mapped.csv", index=False)
    print("Saved stefan solution mapping to B1B3_stefan_mapped.csv")
