"""
classify_Z2L_Z2R_Z2.py

Classification of the Z2L x Z2R x Z2 T-folds on the SO(12) lattice from scratch:

  1. enumerate every choice of the parameters defining the three twist basis
     vectors,

         B1     = [b1 + n.e + N.ebar]        n = (n12, 0^4)
         B1bar  = [b1bar + k.ebar + K.e]     k = (k12, 0^4)   (= nbar, Nbar)
         B_{22bar} = [b22bar + m.e + m.ebar] m = (0^2, m3456), symmetric

     and keep those for which {1, S, Sbar, B1, B1bar, B_{22bar}} satisfies the
     modular invariance conditions (3.4). The conditions are imposed directly
     on the vectors.

  2. reduce the surviving configurations modulo the equivalence relations
     E1-E3 of section 4.1, by an exhaustive search over G_L x G_R (see
     ff_equiv.find_equivalence).  E1 is automatic because configurations are
     compared as additive sets.

No standard-form restrictions (A1-A4, S1-S2 of section 4.3) are imposed: the
full parameter range is scanned and the redundancy is removed by the
equivalence search itself, so the result does not depend on the standard forms
being unique.

Outputs
-------
  Z2L_Z2R_Z2_inequivalent.csv   one row per inequivalent configuration
  Z2L_Z2R_Z2_all_MI.csv         every modular invariant parameter choice,
                                labelled by the class it belongs to
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FF_equivalence_checker_master as FF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "Outputs")


def parameter_scan():
    """All (n12, N) and (k12, K) pairs, and all m3456."""
    bits2 = list(itertools.product((0, 1), repeat=2))
    bits6 = list(itertools.product((0, 1), repeat=6))
    bits4 = list(itertools.product((0, 1), repeat=4))
    # B1^2 = 0 mod 4  <=>  |n12| = |N| mod 4 ; same for B1bar
    left = [(n, N) for n in bits2 for N in bits6 if (sum(n) - sum(N)) % 4 == 0]
    return left, bits4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    lr, ms = parameter_scan()
    print(f"candidate (n12,N) pairs: {len(lr)}   m3456 choices: {len(ms)}")
    print(f"total candidates: {len(lr) * len(lr) * len(ms)}")

    # ---- step 1: modular invariance -------------------------------------
    models = []
    for (n12, N) in lr:
        for (k12, K) in lr:
            for m in ms:
                basis = FF.build_basis_Z2L_Z2R_Z2(n12, N, k12, K, m)
                if FF.modular_invariant(basis):
                    models.append((n12, N, k12, K, m, basis))
    print(f"modular invariant configurations: {len(models)}  "
          f"({time.time()-t0:.0f} s)")

    # ---- step 2: reduce modulo E1-E3 ------------------------------------
    reps = []                     # list of dicts: params, basis, src, fp
    assign = []
    for j, (n12, N, k12, K, m, basis) in enumerate(models):
        xi = FF.additive_set(basis)
        fp = FF.trace_signature(xi)
        tgt = FF.prepare_target_model(xi)
        hit = None
        for r in reps:
            if r["fp"] != fp:
                continue
            if FF.find_equivalence(r["src"], tgt, r["basis"]):
                hit = r["id"]
                break
        if hit is None:
            hit = len(reps)
            reps.append(dict(id=hit, fp=fp, basis=basis,
                             src=FF.prepare_starting_model(xi),
                             params=(n12, N, k12, K, m),
                             twists=[basis[3], basis[4], basis[5]]))
        assign.append(hit)
        if (j + 1) % 250 == 0:
            print(f"  {j+1}/{len(models)} scanned, {len(reps)} classes so far "
                  f"({time.time()-t0:.0f} s)")
    print(f"\ninequivalent configurations: {len(reps)}  ({time.time()-t0:.0f} s)")

    names = {r["id"]: f"class_{r['id']+1:02d}" for r in reps}

    # ---- write ----------------------------------------------------------
    def prow(p):
        n12, N, k12, K, m = p
        d = {"n1": n12[0], "n2": n12[1]}
        d.update({f"N{i+1}": N[i] for i in range(6)})
        d.update({f"K{i+1}": K[i] for i in range(6)})
        d.update({"k1": k12[0], "k2": k12[1]})
        d.update({f"m{i+3}": m[i] for i in range(4)})
        return d

    out = []
    for r in reps:
        d = prow(r["params"])
        d["Label"] = names[r["id"]]
        d["TwistBasis"] = " ; ".join(FF.twist_label(t) for t in r["twists"])
        d["NumberOfMIParameterChoices"] = assign.count(r["id"])
        out.append(d)
    f1 = os.path.join(args.out_dir, "Z2L_Z2R_Z2_inequivalent.csv")
    pd.DataFrame(out).to_csv(f1, index=False)

    allrows = []
    for (params, cid) in zip([m[:5] for m in models], assign):
        d = prow(params)
        d["Class"] = names[cid]
        allrows.append(d)
    f2 = os.path.join(args.out_dir, "Z2L_Z2R_Z2_all_MI.csv")
    pd.DataFrame(allrows).to_csv(f2, index=False)

    print(f"\nwritten:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()