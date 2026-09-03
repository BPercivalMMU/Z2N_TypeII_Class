"""
classify_Z2L_2_Z2R_2.py

Classification of the Z2L^2 x Z2R^2 T-folds on the SO(12) lattice from scratch.

  1. enumerate every choice of the parameters defining the four twist basis
     vectors

         B1    = [b1    + n.e + N.ebar]     n = (n12, 0^4)
         B2    = [b2    + m.e + M.ebar]     m = (0^2, m3456)
         B1bar = [b1bar + k.ebar + K.e]     k = nbar12 , K = Nbar
         B2bar = [b2bar + l.ebar + L.e]     l = mbar3456 , L = Mbar

     and keep those for which {1, S, Sbar, B1, B2, B1bar, B2bar} satisfies the
     modular invariance conditions (3.4). As in the Z2L x Z2R x Z2 script the
     conditions are imposed on the vectors themselves, including the two-loop
     (four-fold overlap) condition.

     The scan is organised as (one-sided data) x (one-sided data):
       * conditions involving only {1,S,Sbar,B1,B2} are applied first, leaving
         the admissible (n12, N, m3456, M);  likewise on the barred side;
       * the 34 conditions mixing the two sides (four inner products
         B_i.B_jbar and thirty four-fold overlaps) are bilinear in the two
         halves, so for each left-hand datum they are applied to all right-hand
         data at once with numpy, filtering progressively.

  2. reduce modulo E1-E3 with the exhaustive G_L x G_R search of ff_equiv.

No standard-form restrictions (A1-A4 of section 4.3), no |N| <= 3 cut and no
gauge fixing of n2 or m4 are imposed; the redundancy is removed by the
equivalence search itself.

    python classify_Z2L_2_Z2R_2.py [--limit-left N]

--limit-left restricts step 1 to the first N left-hand data, for a quick smoke
test; omit it for the full run.
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

BITS2 = list(itertools.product((0, 1), repeat=2))
BITS4 = list(itertools.product((0, 1), repeat=4))
BITS6 = list(itertools.product((0, 1), repeat=6))
ONE, S, SB = FF.make_one(), FF.make_S(), FF.make_Sbar()


def one_sided(barred: bool):
    """(n12, N, m3456, M) passing every condition that involves one side only."""
    out = []
    for n in BITS2:
        for N in BITS6:
            if (sum(n) - sum(N)) % 4:
                continue
            for m in BITS4:
                for M in BITS6:
                    if barred:
                        b1, b2 = FF.make_Bb1(n[0], n[1], N), FF.make_Bb2(m, M)
                    else:
                        b1, b2 = FF.make_B1(n[0], n[1], N), FF.make_B2(m, M)
                    if FF.modular_invariant(np.vstack([ONE, S, SB, b1, b2])):
                        out.append((n, N, m, M, b1, b2))
    return out


def cross_filter(a, n_right, MB1, MB2, MB12):
    """Indices of right-hand data compatible with the left-hand datum a."""
    B1, B2 = a[4], a[5]
    s1 = np.concatenate([B1[:20].astype(np.int32), -B1[20:].astype(np.int32)])
    s2 = np.concatenate([B2[:20].astype(np.int32), -B2[20:].astype(np.int32)])

    idx = np.arange(n_right)
    for M, s in ((MB1, s1), (MB2, s2), (MB2, s1), (MB1, s2)):
        idx = idx[(M[idx] @ s) % 4 == 0]            # B_i . B_jbar = 0 mod 2
        if idx.size == 0:
            return idx

    elems = [ONE, S, SB, B1, B2]
    for tri in itertools.combinations(range(5), 3):        # one barred vector
        p = np.prod([elems[j] for j in tri], axis=0).astype(np.int32)
        for M in (MB1, MB2):
            idx = idx[(M[idx] @ p) % 2 == 0]
            if idx.size == 0:
                return idx
    for pair in itertools.combinations(range(5), 2):       # two barred vectors
        q = np.prod([elems[j] for j in pair], axis=0).astype(np.int32)
        idx = idx[(MB12[idx] @ q) % 2 == 0]
        if idx.size == 0:
            return idx
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit-left", type=int, default=0)
    ap.add_argument("--no-all-mi", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    left = one_sided(False)
    right = one_sided(True)
    print(f"one-sided data: {len(left)} unbarred, {len(right)} barred "
          f"({time.time()-t0:.0f} s)")

    MB1 = np.stack([b[4] for b in right]).astype(np.int32)
    MB2 = np.stack([b[5] for b in right]).astype(np.int32)
    MB12 = MB1 * MB2

    todo = left[:args.limit_left] if args.limit_left else left

    reps, models, assign = [], [], []
    for ia, a in enumerate(todo):
        for ib in cross_filter(a, len(right), MB1, MB2, MB12):
            b = right[ib]
            basis = np.vstack([ONE, S, SB, a[4], a[5], b[4], b[5]]).astype(np.uint8)
            params = (a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])
            xi = FF.additive_set(basis)
            fp = FF.trace_signature(xi)
            tgt = FF.prepare_target_model(xi)
            hit = None
            for r in reps:
                if r["fp"] == fp and FF.find_equivalence(r["src"], tgt, r["basis"]):
                    hit = r["id"]
                    break
            if hit is None:
                hit = len(reps)
                reps.append(dict(id=hit, fp=fp, basis=basis,
                                 src=FF.prepare_starting_model(xi, basis), params=params,
                                 twists=list(basis[3:])))
            models.append(params)
            assign.append(hit)
        if (ia + 1) % 100 == 0:
            print(f"  left datum {ia+1}/{len(todo)}: {len(models)} modular "
                  f"invariant so far, {len(reps)} classes ({time.time()-t0:.0f} s)")

    print(f"\nmodular invariant configurations: {len(models)}")
    print(f"inequivalent configurations: {len(reps)}   ({time.time()-t0:.0f} s)")

    names = {r["id"]: f"class_{r['id']+1:02d}" for r in reps}

    def prow(p):
        n, N, m, M, k, K, l, L = p
        return {"n": str(tuple(n)), "m": str(tuple(m)), "k": str(tuple(k)),
                "l": str(tuple(l)), "N": str(tuple(N)), "M": str(tuple(M)),
                "K": str(tuple(K)), "L": str(tuple(L))}

    rows = []
    for r in reps:
        d = prow(r["params"])
        d["Label"] = names[r["id"]]
        d["TwistBasis"] = " ; ".join(FF.twist_label(t) for t in r["twists"])
        d["NumberOfMIParameterChoices"] = assign.count(r["id"])
        rows.append(d)
    f1 = os.path.join(args.out_dir, "Z2L_2_Z2R_2_inequivalent.csv")
    pd.DataFrame(rows).to_csv(f1, index=False)
    print(f"\nwritten: {f1}")

    if not args.no_all_mi:
        allr = []
        for p, cid in zip(models, assign):
            d = prow(p)
            d["Class"] = names[cid]
            allr.append(d)
        f2 = os.path.join(args.out_dir, "Z2L_2_Z2R_2_all_MI.csv")
        pd.DataFrame(allr).to_csv(f2, index=False)
        print(f"written: {f2}")


if __name__ == "__main__":
    main()