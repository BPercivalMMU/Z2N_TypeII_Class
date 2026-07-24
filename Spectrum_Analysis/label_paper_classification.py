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
    b1 = label_i_iii(row["n1"], row["n2"])
    bb1 = label_i_iii(row["k1"], row["k2"])
    return f"{b1}-{bb1}"


def label_Z2L_Z2(row) -> str:
    b1 = label_i_iii(row["n1"], row["n2"])
    ae = label_A_E(row["m3"], row["m4"], row["m5"], row["m6"])
    return f"{b1}-{ae}"


def label_Z2L_Z2R_Z2(row) -> str:
    b1 = label_i_iii(row["n1"], row["n2"])
    bb1 = label_i_iii(row["k1"], row["k2"])
    ae = label_A_E(row["m3"], row["m4"], row["m5"], row["m6"])
    return f"{b1}-{bb1}-{ae}"


def label_Z2L_2_Z2R(row) -> str:
    n12 = _parse_tuple_cell(row["n12"])
    m = _parse_tuple_cell(row["m3456"])
    k12 = _parse_tuple_cell(row["k12"])
    ivi = label_I_VI(n12[0], n12[1], m[0], m[1], m[2], m[3])
    bb1 = label_i_iii(k12[0], k12[1])
    return f"{ivi}-{bb1}"


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
