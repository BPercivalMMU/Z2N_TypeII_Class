"""
pointgroup_specs.py

Per-point-group data: basis construction, the full parameter range, the input
file reader and the table 15 transcription.
"""
import ast
import itertools

import numpy as np
import pandas as pd

import FF_equivalence_checker_master as FF

BITS2 = list(itertools.product((0, 1), repeat=2))
BITS4 = list(itertools.product((0, 1), repeat=4))
BITS6 = list(itertools.product((0, 1), repeat=6))
ONE, S, SB = FF.make_one(), FF.make_S(), FF.make_Sbar()


def _read(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


_i = lambda r, c: int(str(r[c]).strip())
_t = lambda x: tuple(int(v) for v in ast.literal_eval(str(x).strip()))


# ==========================================================================
#  Z2L x Z2      basis {1, S, Sbar, B1, B_{2 2bar}}
# ==========================================================================
class Z2L_Z2:
    name = "Z2L x Z2"
    basis_names = ["1", "S", "Sb", "B1", "B2b2"]

    @staticmethod
    def build(n12, N, m3456):
        return np.vstack([ONE, S, SB,
                          FF.make_B1(n12[0], n12[1], N),
                          FF.make_B2b2(m3456)]).astype(np.uint8) % 2

    @staticmethod
    def params():
        """Full range.  Only B1^2 = 0 mod 4 is used to keep the loop small."""
        for n in BITS2:
            for N in BITS6:
                if (sum(n) - sum(N)) % 4:
                    continue
                for m in BITS4:
                    yield (n, N, m)

    @staticmethod
    def param_cols(n12, N, m3456):
        d = {"n1": n12[0], "n2": n12[1]}
        d.update({f"m{i+3}": m3456[i] for i in range(4)})
        d.update({f"N{i+1}": N[i] for i in range(6)})
        return d

    @staticmethod
    def read(path):
        df = _read(path)
        bases = [Z2L_Z2.build([_i(r, "n1"), _i(r, "n2")],
                              [_i(r, f"N{j}") for j in range(1, 7)],
                              [_i(r, f"m{j}") for j in range(3, 7)])
                 for _, r in df.iterrows()]
        return df, bases

    table = {
     "i-A":("b1","b2`2"), "ii-A.1":("b1 + e1`1","b2`2"), "ii-A.2":("b1 + e1`5","b2`2"),
     "iii-A.1":("b1 + e12`1`2","b2`2"), "iii-A.2":("b1 + e12`1`5","b2`2"),
     "iii-A.3":("b1 + e12`5`6","b2`2"),
     "ii-B":("b1 + e1`3","b2`2 + e5`5"), "iii-B.1":("b1 + e12`1`3","b2`2 + e5`5"),
     "iii-B.2":("b1 + e12`3`5","b2`2 + e5`5"), "iii-B.3":("b1 + e12`3`6","b2`2 + e5`5"),
     "ii-C":("b1 + e1`3","b2`2 + e3`3"), "iii-C.1":("b1 + e12`1`3","b2`2 + e3`3"),
     "iii-C.2":("b1 + e12`3`5","b2`2 + e3`3"),
     "iii-D":("b1 + e12`3`4","b2`2 + e56`5`6"),
     "i-E":("b1","b2`2 + e35`3`5"), "ii-E.1":("b1 + e1`1","b2`2 + e35`3`5"),
     "ii-E.2":("b1 + e1`5","b2`2 + e35`3`5"), "ii-E.3":("b1 + e1`6","b2`2 + e35`3`5"),
     "iii-E.1":("b1 + e12`1`2","b2`2 + e35`3`5"), "iii-E.2":("b1 + e12`1`5","b2`2 + e35`3`5"),
     "iii-E.3":("b1 + e12`1`6","b2`2 + e35`3`5"), "iii-E.4":("b1 + e12`3`4","b2`2 + e35`3`5"),
     "iii-E.5":("b1 + e12`5`6","b2`2 + e35`3`5"),
    }


# ==========================================================================
#  Z2L^2 x Z2R   basis {1, S, Sbar, B1, B2, B1bar}
# ==========================================================================
class Z2L_2_Z2R:
    name = "Z2L^2 x Z2R"
    basis_names = ["1", "S", "Sb", "B1", "B2", "B1b"]

    @staticmethod
    def build(n12, N, m3456, M, k12, K):
        return np.vstack([ONE, S, SB,
                          FF.make_B1(n12[0], n12[1], N),
                          FF.make_B2(m3456, M),
                          FF.make_Bb1(k12[0], k12[1], K)]).astype(np.uint8) % 2

    @staticmethod
    def params():
        """
        Full range, organised as (one-sided data) x (one-sided data): the
        conditions involving only {1,S,Sbar,B1,B2} are applied first, and
        likewise for {1,S,Sbar,B1bar}, before the mixed conditions are tested.
        """
        left = [(n, N, m, M) for n in BITS2 for N in BITS6
                if (sum(n) - sum(N)) % 4 == 0
                for m in BITS4 for M in BITS6
                if FF.modular_invariant(np.vstack([ONE, S, SB,
                                                   FF.make_B1(n[0], n[1], N),
                                                   FF.make_B2(m, M)]))]
        right = [(k, K) for k in BITS2 for K in BITS6
                 if (sum(k) - sum(K)) % 4 == 0
                 and FF.modular_invariant(np.vstack([ONE, S, SB,
                                                     FF.make_Bb1(k[0], k[1], K)]))]
        for (n, N, m, M) in left:
            for (k, K) in right:
                yield (n, N, m, M, k, K)

    @staticmethod
    def param_cols(n12, N, m3456, M, k12, K):
        return {"n12": str(tuple(n12)), "m3456": str(tuple(m3456)),
                "k12": str(tuple(k12)), "N": str(tuple(N)),
                "M": str(tuple(M)), "K": str(tuple(K))}

    @staticmethod
    def read(path):
        df = _read(path)
        bases = [Z2L_2_Z2R.build(_t(r["n12"]), _t(r["N"]), _t(r["m3456"]),
                                 _t(r["M"]), _t(r["k12"]), _t(r["K"]))
                 for _, r in df.iterrows()]
        return df, bases

    table = {
     "I-i":("b1","b2","b`1"), "II-i":("b1","b2 + e35`1","b`1"),
     "III-i":("b1","b2 + e3456`1`2","b`1"), "IV-i":("b1 + e1`1","b2 + e356`2","b`1"),
     "II-ii":("b1","b2 + e35`3","b`1 + e1`1"),
     "III-ii":("b1","b2 + e3456`2`4","b`1 + e1`1"),
     "IV-ii.1":("b1 + e1`2","b2 + e356`3","b`1 + e2`1"),
     "IV-ii.2":("b1 + e1`3","b2 + e356`2","b`1 + e4`1"),
     "IV-ii.3":("b1 + e1`3","b2 + e356`4","b`1 + e5`1"),
     "V-ii.1":("b1 + e12`2`3","b2 + e3456`2`4","b`1 + e5`1"),
     "V-ii.2":("b1 + e12`2`3","b2 + e3456`3`4","b`1 + e3`1"),
     "V-ii.3":("b1 + e12`3`4","b2 + e3456`2`3","b`1 + e1`1"),
     "II-iii":("b1","b2 + e35`1","b`1 + e12`1`2"),
     "III-iii":("b1","b2 + e3456`3`4","b`1 + e12`1`2"),
     "IV-iii.1":("b1 + e1`1","b2 + e356`2","b`1 + e56`1`2"),
     "IV-iii.2":("b1 + e1`1","b2 + e356`3","b`1 + e45`1`2"),
     "IV-iii.3":("b1 + e1`3","b2 + e356`1","b`1 + e25`1`2"),
     "IV-iii.4":("b1 + e1`3","b2 + e356`4","b`1 + e24`1`2"),
     "V-iii.1":("b1 + e12`1`3","b2 + e3456`3`4","b`1 + e15`1`2"),
     "V-iii.2":("b1 + e12`3`4","b2 + e3456`1`3","b`1 + e35`1`2"),
     "V-iii.3":("b1 + e12`1`3","b2 + e3456`1`4","b`1 + e13`1`2"),
     "VI-iii.1":("b1 + e12`1`2","b2 + e34`3`4","b`1 + e12`1`2"),
     "VI-iii.2":("b1 + e12`3`4","b2 + e34`1`2","b`1 + e34`1`2"),
     "VI-iii.3":("b1 + e12`3`4","b2 + e34`1`5","b`1 + e35`1`2"),
     "VI-iii.4":("b1 + e12`1`3","b2 + e34`2`4","b`1 + e13`1`2"),
     "VI-iii.5":("b1 + e12`1`3","b2 + e34`4`5","b`1 + e15`1`2"),
     "VI-iii.6":("b1 + e12`3`4","b2 + e34`5`6","b`1 + e56`1`2"),
    }


# ==========================================================================
#  Z2L^2         basis {1, S, Sbar, B1, B2}
# ==========================================================================
class Z2L_2:
    name = "Z2L^2"
    basis_names = ["1", "S", "Sb", "B1", "B2"]

    @staticmethod
    def build(n12, N, m3456, M):
        return np.vstack([ONE, S, SB,
                          FF.make_B1(n12[0], n12[1], N),
                          FF.make_B2(m3456, M)]).astype(np.uint8) % 2

    @staticmethod
    def params():
        """Full range.  Only B1^2 = 0 mod 4 is used to keep the loop small."""
        for n in BITS2:
            for N in BITS6:
                if (sum(n) - sum(N)) % 4:
                    continue
                for m in BITS4:
                    for M in BITS6:
                        yield (n, N, m, M)

    @staticmethod
    def param_cols(n12, N, m3456, M):
        return {"n": str(tuple(n12)), "N": str(tuple(N)),
                "m": str(tuple(m3456)), "M": str(tuple(M))}

    table = {
     "I": ("b1", "b2"),
     "II": ("b1", "b2 + e35`1"),
     "III": ("b1", "b2 + e3456`1`2"),
     "IV": ("b1 + e1`1", "b2 + e356`2"),
     "V": ("b1 + e12`1`2", "b2 + e3456`1`3"),
     "VI": ("b1 + e12`1`2", "b2 + e34`3`4"),
    }


# ==========================================================================
#  Z2L x Z2R     basis {1, S, Sbar, B1, B1bar}
# ==========================================================================
class Z2L_Z2R:
    name = "Z2L x Z2R"
    basis_names = ["1", "S", "Sb", "B1", "B1b"]

    @staticmethod
    def build(n12, N, k12, K):
        return np.vstack([ONE, S, SB,
                          FF.make_B1(n12[0], n12[1], N),
                          FF.make_Bb1(k12[0], k12[1], K)]).astype(np.uint8) % 2

    @staticmethod
    def params():
        """
        Full range, organised as (one-sided data) x (one-sided data): each
        side's own self modular-invariance condition is applied first, before
        the cross conditions between B1 and B1bar are tested.
        """
        side = [(n, N) for n in BITS2 for N in BITS6 if (sum(n) - sum(N)) % 4 == 0]
        for (n12, N) in side:
            for (k12, K) in side:
                yield (n12, N, k12, K)

    @staticmethod
    def param_cols(n12, N, k12, K):
        return {"n": str(tuple(n12)), "N": str(tuple(N)),
                "k": str(tuple(k12)), "K": str(tuple(K))}

    table = {
     "i-i": ("b1", "b`1"),
     "ii-i": ("b1 + e1`1", "b`1"),
     "i-ii": ("b1", "b`1 + e1`1"),
     "iii-i": ("b1 + e12`1`2", "b`1"),
     "i-iii": ("b1", "b`1 + e12`1`2"),
     "ii-ii.1": ("b1 + e1`1", "b`1 + e1`1"),
     "ii-ii.2": ("b1 + e1`2", "b`1 + e2`1"),
     "ii-ii.3": ("b1 + e1`3", "b`1 + e3`1"),
     "iii-ii.1": ("b1 + e12`1`2", "b`1 + e1`1"),
     "ii-iii.1": ("b1 + e1`1", "b`1 + e12`1`2"),
     "iii-ii.2": ("b1 + e12`2`3", "b`1 + e3`1"),
     "ii-iii.2": ("b1 + e1`3", "b`1 + e23`1`2"),
     "iii-ii.3": ("b1 + e12`3`4", "b`1 + e1`1"),
     "ii-iii.3": ("b1 + e1`1", "b`1 + e34`1`2"),
     "iii-iii.1": ("b1 + e12`1`2", "b`1 + e12`1`2"),
     "iii-iii.2": ("b1 + e12`1`3", "b`1 + e13`1`2"),
     "iii-iii.3": ("b1 + e12`3`4", "b`1 + e34`1`2"),
    }


SPECS = {"Z2L_Z2": Z2L_Z2, "Z2L_2_Z2R": Z2L_2_Z2R, "Z2L_2": Z2L_2, "Z2L_Z2R": Z2L_Z2R}
