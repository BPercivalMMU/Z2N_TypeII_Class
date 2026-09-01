"""
classify_Z2L_Z2R.py

Classification of the Z2L x Z2R T-folds on the SO(12) lattice from scratch:

  1. enumerate the full range of the parameters defining the twist basis
     vectors and keep those for which the basis satisfies the modular
     invariance conditions (3.4).  The conditions are imposed directly on the
     vectors -- including the two-loop four-fold overlap condition -- so no
     hand-derived parameter relation enters, and no standard-form restriction
     (A1-A4, S1-S2 of section 4.3), no |N| <= 3 cut and no gauge fixing of n2
     or k2 is applied.

  2. reduce the survivors modulo E1-E3 with the exhaustive G_L x G_R search of
     FF_equivalence_checker_master.find_equivalence.

    python classify_Z2L_Z2R.py [--out-dir DIR] [--no-all-mi]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import classification_tools_all as CT
from pointgroup_specs import SPECS

SPEC = SPECS["Z2L_Z2R"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "Outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-all-mi", action="store_true",
                    help="do not write the list of every modular invariant choice")
    args = ap.parse_args()

    res = CT.classify(SPEC.name, SPEC.params(), SPEC.build,
                      SPEC.basis_names, args.out_dir, SPEC.table)
    CT.write_classes(res, os.path.join(args.out_dir, "Z2L_Z2R_inequivalent.csv"),
                     SPEC.param_cols)
    if not args.no_all_mi:
        CT.write_all_MI(res, os.path.join(args.out_dir, "Z2L_Z2R_all_MI.csv"),
                        SPEC.param_cols)


if __name__ == "__main__":
    main()