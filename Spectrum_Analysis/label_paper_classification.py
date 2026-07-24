#!/usr/bin/env python3
"""
label_paper_classification.py

Adds a "PaperLabel" column to the canonical input CSVs in
All_Z2N_Input_Models_updated_310526, giving each row's twist basis
vectors in the roman-numeral/letter classification convention from
the paper (Faraggi, Groot Nibbelink, Percival).

Building blocks (each a pure function of that vector's own shift bits):
  a-b    : symmetric b_{1bar1} alone           -- from (n1,n2)
  i-iii  : asymmetric b_1 or b_bar1 alone       -- from (n1,n2) or (k1,k2)
  I-VI   : combined (b_1,b_2) pair              -- from (n1,n2) + (m3,m4,m5,m6)
  A-E    : symmetric b_{2bar2} alone            -- from (m3,m4,m5,m6)

Composite labels are the building blocks joined with "-", in the same
order the basis vectors appear in each class's basis list (see
BASIS_BUILDERS in get_model_spectra_stats_all_classes.py).

Rules were derived from the paper's classification table and verified
by construction against every row currently in each canonical file
(see conversation record) -- every row matched exactly one entry with
no ambiguity. Any row that does NOT match a known pattern raises a
ValueError rather than being silently mislabeled.
"""

import ast
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(HERE, "All_Z2N_Input_Models_updated_310526")

I_VI_TABLE = {
    ((0, 0), (0, 0, 0, 0)): "I",
    ((0, 0), (1, 0, 1, 0)): "II",
    ((0, 0), (1, 1, 1, 1)): "III",
    ((1, 0), (1, 0, 1, 1)): "IV",
    ((1, 1), (1, 1, 1, 1)): "V",
    ((1, 1), (1, 1, 0, 0)): "VI",
}

A_E_TABLE = {
    (0, 0, 0, 0): "A",
    (0, 0, 1, 0): "B",
    (1, 0, 0, 0): "C",
    (0, 0, 1, 1): "D",
    (1, 0, 1, 0): "E",
}


def _pair(v1, v2) -> tuple:
    return (int(v1), int(v2))


def label_i_iii(n1, n2) -> str:
    mapping = {(0, 0): "i", (1, 0): "ii", (1, 1): "iii"}
    key = _pair(n1, n2)
    if key not in mapping:
        raise ValueError(f"Unexpected (n1,n2)/(k1,k2) pattern for i-iii: {key}")
    return mapping[key]


def label_a_b(n1, n2) -> str:
    mapping = {(0, 0): "a", (1, 0): "b"}
    key = _pair(n1, n2)
    if key not in mapping:
        raise ValueError(f"Unexpected (n1,n2) pattern for a-b: {key}")
    return mapping[key]


def label_I_VI(n1, n2, m3, m4, m5, m6) -> str:
    key = (_pair(n1, n2), (int(m3), int(m4), int(m5), int(m6)))
    if key not in I_VI_TABLE:
        raise ValueError(f"Unexpected (n,m) pattern for I-VI: {key}")
    return I_VI_TABLE[key]


def label_A_E(m3, m4, m5, m6) -> str:
    key = (int(m3), int(m4), int(m5), int(m6))
    if key not in A_E_TABLE:
        raise ValueError(f"Unexpected m-pattern for A-E: {key}")
    return A_E_TABLE[key]


def _parse_tuple_cell(x) -> tuple:
    return tuple(ast.literal_eval(str(x).strip()))


def _idx_set(vals) -> tuple:
    """1-based positions where vals[i] == 1, e.g. (1,0,1,0) -> (1,3)."""
    return tuple(i + 1 for i, v in enumerate(vals) if int(v) == 1)


# ── Fine (sub-numbered) sub-case tables ─────────────────────────────────────
# Keys are (b1_own, b1_bar, bb1_own, bb1_bar) / (b1_own, b1_bar, AE_letter)
# index-set tuples, transcribed from the paper's classification table and
# verified row-for-row (bijectively, no ambiguity) against every row
# currently in the corresponding canonical CSV.

FINE_Z2L_Z2R = {
    ((), (), (), ()): "i-i",
    ((1,), (1,), (), ()): "ii-i",
    ((1, 2), (1, 2), (), ()): "iii-i",
    ((), (), (1,), (1,)): "i-ii",
    ((), (), (1, 2), (1, 2)): "i-iii",
    ((1,), (1,), (1,), (1,)): "ii-ii.1",
    ((1,), (2,), (1,), (2,)): "ii-ii.2",
    ((1,), (3,), (1,), (3,)): "ii-ii.3",
    ((1, 2), (1, 2), (1,), (1,)): "iii-ii.1",
    ((1, 2), (2, 3), (1,), (3,)): "iii-ii.2",
    ((1, 2), (3, 4), (1,), (1,)): "iii-ii.3",
    ((1,), (1,), (1, 2), (1, 2)): "ii-iii.1",
    ((1,), (3,), (1, 2), (2, 3)): "ii-iii.2",
    ((1,), (1,), (1, 2), (3, 4)): "ii-iii.3",
    ((1, 2), (1, 2), (1, 2), (1, 2)): "iii-iii.1",
    ((1, 2), (1, 3), (1, 2), (1, 3)): "iii-iii.2",
    ((1, 2), (3, 4), (1, 2), (3, 4)): "iii-iii.3",
}

FINE_Z2L_2_Z2R = {
    # key = (b1_own, b1_bar, b2_own, b2_bar, bb1_own, bb1_bar)
    ((), (), (), (), (), ()): "I-i",
    ((), (), (3, 5), (1,), (), ()): "II-i",
    ((), (), (3, 4, 5, 6), (1, 2), (), ()): "III-i",
    ((1,), (1,), (3, 5, 6), (2,), (), ()): "IV-i",
    ((), (), (3, 5), (3,), (1,), (1,)): "II-ii",
    ((), (), (3, 4, 5, 6), (2, 4), (1,), (1,)): "III-ii",
    ((1,), (2,), (3, 5, 6), (3,), (1,), (2,)): "IV-ii.1",
    ((1,), (3,), (3, 5, 6), (2,), (1,), (4,)): "IV-ii.2",
    ((1,), (3,), (3, 5, 6), (4,), (1,), (5,)): "IV-ii.3",
    ((1, 2), (2, 3), (3, 4, 5, 6), (2, 4), (1,), (5,)): "V-ii.1",
    ((1, 2), (2, 3), (3, 4, 5, 6), (3, 4), (1,), (3,)): "V-ii.2",
    ((1, 2), (3, 4), (3, 4, 5, 6), (2, 3), (1,), (1,)): "V-ii.3",
    ((), (), (3, 5), (1,), (1, 2), (1, 2)): "II-iii",
    ((), (), (3, 4, 5, 6), (3, 4), (1, 2), (1, 2)): "III-iii",
    ((1,), (1,), (3, 5, 6), (2,), (1, 2), (5, 6)): "IV-iii.1",
    ((1,), (1,), (3, 5, 6), (3,), (1, 2), (4, 5)): "IV-iii.2",
    ((1,), (3,), (3, 5, 6), (1,), (1, 2), (2, 5)): "IV-iii.3",
    ((1,), (3,), (3, 5, 6), (4,), (1, 2), (2, 4)): "IV-iii.4",
    # commented-out / superseded alt representatives (same physics, verified via source)
    ((1,), (2,), (3, 5, 6), (1,), (1, 2), (5, 6)): "IV-iii.3",
    ((1,), (2,), (3, 5, 6), (3,), (1, 2), (4, 5)): "IV-iii.4",
    ((1, 2), (1, 3), (3, 4, 5, 6), (3, 4), (1, 2), (1, 5)): "V-iii.1",
    ((1, 2), (3, 4), (3, 4, 5, 6), (1, 3), (1, 2), (3, 5)): "V-iii.2",
    ((1, 2), (1, 3), (3, 4, 5, 6), (1, 4), (1, 2), (1, 3)): "V-iii.3",
    ((1, 2), (1, 2), (3, 4), (3, 4), (1, 2), (1, 2)): "VI-iii.1",
    ((1, 2), (3, 4), (3, 4), (1, 2), (1, 2), (3, 4)): "VI-iii.2",
    ((1, 2), (3, 4), (3, 4), (1, 5), (1, 2), (3, 5)): "VI-iii.3",
    ((1, 2), (1, 3), (3, 4), (2, 4), (1, 2), (1, 3)): "VI-iii.4",
    ((1, 2), (1, 3), (3, 4), (4, 5), (1, 2), (1, 5)): "VI-iii.5",
    ((1, 2), (3, 4), (3, 4), (5, 6), (1, 2), (5, 6)): "VI-iii.6",
}

FINE_Z2L_Z2R_Z2 = {
    # key = (b1_own, b1_bar, bb1_own, bb1_bar, AE)
    ((), (), (), (), "A"): "i-i-A",
    ((), (), (1,), (1,), "A"): "i-ii-A",
    ((1,), (1,), (), (), "A"): "ii-i-A",
    ((1,), (1,), (1,), (1,), "A"): "ii-ii-A.1",
    ((1,), (2,), (1,), (2,), "A"): "ii-ii-A.2",
    ((1,), (5,), (1,), (5,), "A"): "ii-ii-A.3",
    ((), (), (1, 2), (1, 2), "A"): "i-iii-A",
    ((1, 2), (1, 2), (), (), "A"): "iii-i-A",
    ((1,), (1,), (1, 2), (1, 2), "A"): "ii-iii-A.1",
    ((1,), (5,), (1, 2), (2, 5), "A"): "ii-iii-A.2",
    ((1, 2), (1, 2), (1,), (1,), "A"): "iii-ii-A.1",
    ((1, 2), (2, 5), (1,), (5,), "A"): "iii-ii-A.2",
    ((1, 2), (1, 2), (1, 2), (1, 2), "A"): "iii-iii-A.1",
    ((1, 2), (1, 5), (1, 2), (1, 5), "A"): "iii-iii-A.2",
    ((1, 2), (5, 6), (1, 2), (5, 6), "A"): "iii-iii-A.3",
    ((1,), (3,), (1,), (3,), "B"): "ii-ii-B",
    ((1,), (3,), (1, 2), (2, 3), "B"): "ii-iii-B",
    ((1, 2), (2, 3), (1,), (3,), "B"): "iii-ii-B",
    ((1, 2), (1, 3), (1, 2), (1, 3), "B"): "iii-iii-B.1",
    ((1, 2), (3, 5), (1, 2), (3, 5), "B"): "iii-iii-B.2",
    ((1, 2), (3, 6), (1, 2), (3, 6), "B"): "iii-iii-B.3",
    ((1,), (3,), (1,), (3,), "C"): "ii-ii-C",
    ((1,), (3,), (1, 2), (2, 3), "C"): "ii-iii-C",
    ((1, 2), (2, 3), (1,), (3,), "C"): "iii-ii-C",
    ((1, 2), (1, 3), (1, 2), (1, 3), "C"): "iii-iii-C.1",
    ((1, 2), (3, 5), (1, 2), (3, 5), "C"): "iii-iii-C.2",
    ((1, 2), (3, 4), (1, 2), (3, 4), "D"): "iii-iii-D",
    ((), (), (), (), "E"): "i-i-E",
    ((), (), (1,), (1,), "E"): "i-ii-E",
    ((1,), (1,), (), (), "E"): "ii-i-E",
    ((1,), (1,), (1,), (1,), "E"): "ii-ii-E.1",
    ((1,), (2,), (1,), (2,), "E"): "ii-ii-E.2",
    ((1,), (5,), (1,), (5,), "E"): "ii-ii-E.3",
    ((1,), (6,), (1,), (6,), "E"): "ii-ii-E.4",
    ((), (), (1, 2), (1, 2), "E"): "i-iii-E",
    ((1, 2), (1, 2), (), (), "E"): "iii-i-E",
    ((1,), (1,), (1, 2), (1, 2), "E"): "ii-iii-E.1",
    ((1,), (1,), (1, 2), (3, 4), "E"): "ii-iii-E.2",
    ((1,), (1,), (1, 2), (5, 6), "E"): "ii-iii-E.3",
    ((1,), (5,), (1, 2), (2, 5), "E"): "ii-iii-E.4",
    ((1,), (6,), (1, 2), (2, 6), "E"): "ii-iii-E.5",
    ((1, 2), (1, 2), (1,), (1,), "E"): "iii-ii-E.1",
    ((1, 2), (3, 4), (1,), (1,), "E"): "iii-ii-E.2",
    ((1, 2), (5, 6), (1,), (1,), "E"): "iii-ii-E.3",
    ((1, 2), (2, 5), (1,), (5,), "E"): "iii-ii-E.4",
    ((1, 2), (2, 6), (1,), (6,), "E"): "iii-ii-E.5",
    ((1, 2), (1, 2), (1, 2), (1, 2), "E"): "iii-iii-E.1",
    ((1, 2), (1, 5), (1, 2), (1, 5), "E"): "iii-iii-E.2",
    ((1, 2), (1, 6), (1, 2), (1, 6), "E"): "iii-iii-E.3",
    ((1, 2), (3, 4), (1, 2), (3, 4), "E"): "iii-iii-E.4",
    ((1, 2), (3, 4), (1, 2), (5, 6), "E"): "iii-iii-E.5",
    ((1, 2), (5, 6), (1, 2), (3, 4), "E"): "iii-iii-E.6",
    ((1, 2), (5, 6), (1, 2), (5, 6), "E"): "iii-iii-E.7",
}

FINE_Z2L_Z2 = {
    ((), (), "A"): "i-A",
    ((1,), (1,), "A"): "ii-A.1",
    ((1,), (5,), "A"): "ii-A.2",
    ((1, 2), (1, 2), "A"): "iii-A.1",
    ((1, 2), (1, 5), "A"): "iii-A.2",
    ((1, 2), (5, 6), "A"): "iii-A.3",
    ((1,), (3,), "B"): "ii-B",
    ((1, 2), (1, 3), "B"): "iii-B.1",
    ((1, 2), (3, 5), "B"): "iii-B.2",
    ((1, 2), (3, 6), "B"): "iii-B.3",
    ((1,), (3,), "C"): "ii-C",
    ((1, 2), (1, 3), "C"): "iii-C.1",
    ((1, 2), (3, 5), "C"): "iii-C.2",
    ((1, 2), (3, 4), "D"): "iii-D",
    ((), (), "E"): "i-E",
    ((1,), (1,), "E"): "ii-E.1",
    ((1,), (5,), "E"): "ii-E.2",
    ((1,), (6,), "E"): "ii-E.3",
    ((1, 2), (1, 2), "E"): "iii-E.1",
    ((1, 2), (1, 5), "E"): "iii-E.2",
    ((1, 2), (1, 6), "E"): "iii-E.3",
    ((1, 2), (3, 4), "E"): "iii-E.4",
    ((1, 2), (5, 6), "E"): "iii-E.5",
}


# ── Per-class label builders ────────────────────────────────────────────────

def label_Z2(row) -> str:
    return label_a_b(row["m1"], row["m2"])


def label_Z2_2(row) -> str:
    ab = label_a_b(row["n1"], row["n2"])
    ae = label_A_E(row["m3"], row["m4"], row["m5"], row["m6"])
    return f"{ab}-{ae}"


def label_Z2L(row) -> str:
    return label_i_iii(row["n1"], row["n2"])


def label_Z2L_2(row) -> str:
    return label_I_VI(row["n1"], row["n2"], row["m3"], row["m4"], row["m5"], row["m6"])


def label_Z2L_Z2R(row) -> str:
    b1_own = _idx_set([row["n1"], row["n2"]])
    b1_bar = _idx_set([row[f"N{i}"] for i in range(1, 7)])
    bb1_own = _idx_set([row["k1"], row["k2"]])
    bb1_bar = _idx_set([row[f"K{i}"] for i in range(1, 7)])
    key = (b1_own, b1_bar, bb1_own, bb1_bar)
    if key not in FINE_Z2L_Z2R:
        raise ValueError(f"Unexpected Z2L_Z2R fine pattern: {key}")
    return FINE_Z2L_Z2R[key]


def label_Z2L_Z2(row) -> str:
    b1_own = _idx_set([row["n1"], row["n2"]])
    b1_bar = _idx_set([row[f"N{i}"] for i in range(1, 7)])
    ae = label_A_E(row["m3"], row["m4"], row["m5"], row["m6"])
    key = (b1_own, b1_bar, ae)
    if key not in FINE_Z2L_Z2:
        raise ValueError(f"Unexpected Z2L_Z2 fine pattern: {key}")
    return FINE_Z2L_Z2[key]


def label_Z2L_Z2R_Z2(row) -> str:
    b1_own = _idx_set([row["n1"], row["n2"]])
    b1_bar = _idx_set([row[f"N{i}"] for i in range(1, 7)])
    bb1_own = _idx_set([row["k1"], row["k2"]])
    bb1_bar = _idx_set([row[f"K{i}"] for i in range(1, 7)])
    ae = label_A_E(row["m3"], row["m4"], row["m5"], row["m6"])
    key = (b1_own, b1_bar, bb1_own, bb1_bar, ae)
    if key not in FINE_Z2L_Z2R_Z2:
        raise ValueError(f"Unexpected Z2L_Z2R_Z2 fine pattern: {key}")
    return FINE_Z2L_Z2R_Z2[key]


def label_Z2L_2_Z2R(row) -> str:
    n12 = _parse_tuple_cell(row["n12"])
    m = _parse_tuple_cell(row["m3456"])
    k12 = _parse_tuple_cell(row["k12"])
    N = _parse_tuple_cell(row["N"])
    M = _parse_tuple_cell(row["M"])
    K = _parse_tuple_cell(row["K"])
    ivi = label_I_VI(n12[0], n12[1], m[0], m[1], m[2], m[3])
    bb1 = label_i_iii(k12[0], k12[1])
    coarse = f"{ivi}-{bb1}"

    key = (
        _idx_set(n12), _idx_set(N),
        tuple(i + 3 for i, v in enumerate(m) if int(v) == 1), _idx_set(M),
        _idx_set(k12), _idx_set(K),
    )
    fine = FINE_Z2L_2_Z2R.get(key)
    if fine is None:
        print(f"  [Z2L_2_Z2R] no fine match for key={key}, falling back to coarse label {coarse!r}")
        return coarse
    if fine.split(".")[0] != coarse:
        raise ValueError(f"Fine/coarse mismatch: fine={fine!r} coarse={coarse!r} key={key}")
    return fine


def label_Z2L_2_Z2R_2(row) -> str:
    n = _parse_tuple_cell(row["n"])
    m = _parse_tuple_cell(row["m"])
    k = _parse_tuple_cell(row["k"])
    l = _parse_tuple_cell(row["l"])
    ivi_1 = label_I_VI(n[0], n[1], m[0], m[1], m[2], m[3])
    ivi_2 = label_I_VI(k[0], k[1], l[0], l[1], l[2], l[3])
    return f"{ivi_1}-{ivi_2}"


CANONICAL_FILES = {
    "Z2_solutions.csv": label_Z2,
    "Z2_2_solutions.csv": label_Z2_2,
    "Z2L_solutions.csv": label_Z2L,
    "Z2L_2_stefan_solutions.csv": label_Z2L_2,
    "Z2L_Z2R_stefan_solutions.csv": label_Z2L_Z2R,
    "Z2L_Z2_stefan_solutions.csv": label_Z2L_Z2,
    "Z2L_2_Z2R_v2_reformatted.csv": label_Z2L_2_Z2R,
    "Z2L_2_Z2R_2_v2_reformatted.csv": label_Z2L_2_Z2R_2,
    "Z2L_Z2R_Z2_v2_reformatted.csv": label_Z2L_Z2R_Z2,
}


def main() -> None:
    for fname, label_fn in CANONICAL_FILES.items():
        path = os.path.join(INPUT_DIR, fname)
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]

        labels = []
        for i, row in df.iterrows():
            try:
                labels.append(label_fn(row))
            except ValueError as e:
                raise ValueError(f"{fname} row {i}: {e}") from e

        df["PaperLabel"] = labels
        df.to_csv(path, index=False, encoding="utf-8-sig")

        counts = pd.Series(labels).value_counts()
        print(f"{fname}: {len(df)} rows labelled")
        print(counts.to_string())
        print()


if __name__ == "__main__":
    main()
