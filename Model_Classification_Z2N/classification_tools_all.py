"""
classification_tools.py

Shared code used by the equivalence-check and classify scripts below.

  check_models()  : compares every model in a given list against every other
                     one, to find which pairs are equivalent under E1-E3,
                     then writes the list back out with new columns:
                     Unique / ClassRepresentative / EquivalentTo /
                     RelabellingToRepresentative / EquivalenceNotes.

  classify()      : enumerate the full parameter range, keep the modular
                     invariant choices, and reduce them modulo E1-E3.

Both use FF_equivalence_checker_master.find_equivalence, which searches all of G_L x G_R
(46080^2 relabellings) without enumerating it, and verifies every hit by
explicit equality of the additive sets.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import FF_equivalence_checker_master as FF


# --------------------------------------------------------------------------
def perm_note(g) -> str:
    l, r = FF.relabelling_note(g[0]), FF.relabelling_note(g[1])
    if l == r:
        return f"holomorphic and anti-holomorphic indices both: {l}"
    return f"holomorphic: {l}; anti-holomorphic: {r}"


def basis_note(src_twists, dst_basis, names, g) -> str:
    """Only the generators that must be recombined are listed."""
    changed, same = [], []
    for name, vec in zip(names[3:], src_twists):
        coef = FF.basis_coefficients(dst_basis, FF.apply_g(vec, g[0], g[1]))
        if coef is None:
            return "?"
        terms = [names[j] for j in range(len(coef)) if coef[j]]
        if all(x in terms for x in ("1", "S", "Sb")):
            terms = [x for x in terms if x not in ("1", "S", "Sb")] + ["E"]
        (same if terms == [name] else changed).append(
            name if terms == [name] else f"{name} -> " + " + ".join(terms))
    if not changed:
        return "none needed (the relabelling alone maps one basis onto the other)"
    txt = "; ".join(changed)
    if same:
        txt += f" ({', '.join(same)} unchanged)"
    return txt


def label_lookup(basis: np.ndarray, table: Dict[str, Sequence[str]]) -> str:
    """Match a basis against a table-15 transcription via describe()."""
    key = " ; ".join(FF.twist_label(t) for t in basis[3:])
    inv = {" ; ".join(v): k for k, v in table.items()}
    return inv.get(key, "")


# --------------------------------------------------------------------------
def check_models(name: str, df: pd.DataFrame, bases: List[np.ndarray],
                 labels: List[str], names: List[str], out_path: str) -> pd.DataFrame:
    n = len(bases)
    models = [dict(idx=i, label=labels[i], basis=bases[i],
                   xi=FF.additive_set(bases[i]), twists=list(bases[i][3:]))
              for i in range(n)]

    print(f"read {n} models")
    for m in models:
        assert FF.modular_invariant(m["basis"]), f"{m['label']} is not modular invariant"
    print("all input models are modular invariant")

    # group model indices by their trace_signature: only models sharing a
    # signature can possibly be equivalent, so only those need comparing
    same_signature_groups: Dict[tuple, List[int]] = {}
    for m in models:
        same_signature_groups.setdefault(FF.trace_signature(m["xi"]), []).append(m["idx"])

    # group_leader[i] is model i's current best guess at which model index
    # leads its equivalence group; find() follows these guesses to the
    # group's actual leader, and two models are found equivalent by pointing
    # the higher-numbered one's leader at the lower-numbered one's
    group_leader = list(range(n))

    def find(a):
        while group_leader[a] != a:
            group_leader[a] = group_leader[group_leader[a]]
            a = group_leader[a]
        return a

    for candidates in same_signature_groups.values():
        for a in candidates:
            A = FF.prepare_starting_model(models[a]["xi"], models[a]["basis"])
            for b in candidates:
                if b <= a or find(a) == find(b):
                    continue
                if FF.find_equivalence(A, FF.prepare_target_model(models[b]["xi"])):
                    ra, rb = find(a), find(b)
                    group_leader[max(ra, rb)] = min(ra, rb)

    classes: Dict[int, List[int]] = {}
    for k in range(n):
        classes.setdefault(find(k), []).append(k)

    rep_of, equiv_of, gmap, notes = {}, {}, {}, {}
    for rep, group_members in classes.items():
        for k in group_members:
            rep_of[k] = models[rep]["label"]
            equiv_of[k] = ", ".join(models[j]["label"] for j in group_members if j != k)
            if k == rep:
                others = [models[j]["label"] for j in group_members if j != k]
                notes[k] = ("unique -- no other configuration in the list is "
                            "equivalent to it" if not others else
                            "class representative; equivalent model(s) in this "
                            "list: " + ", ".join(others))
                continue
            # always solved in the direction  k -> rep, then verified
            g = FF.find_equivalence(
                FF.prepare_starting_model(models[k]["xi"], models[k]["basis"]),
                FF.prepare_target_model(models[rep]["xi"]))
            assert g is not None
            tgt = np.concatenate([FF._half_targets(*g[0]),
                                  20 + FF._half_targets(*g[1])])
            img = np.zeros_like(models[k]["xi"])
            img[:, tgt] = models[k]["xi"]
            assert np.array_equal(np.sort(FF.encode(img)),
                                  np.sort(FF.encode(models[rep]["xi"])))
            gmap[k] = f"left: {FF.relabelling_note(g[0])}  |  right: {FF.relabelling_note(g[1])}"
            notes[k] = (f"= {models[rep]['label']}.  "
                        f"E2/E3 permutation -- {perm_note(g)}.  "
                        f"E1 change of basis (E = 1+S+Sb) -- "
                        f"{basis_note(models[k]['twists'], models[rep]['basis'], names, g)}")

    out = df.copy()
    out["PaperLabel"] = labels
    out["TwistBasis"] = [" ; ".join(FF.twist_label(t) for t in m["twists"]) for m in models]
    out["Unique"] = ["yes" if len(classes[find(k)]) == 1 else "no" for k in range(n)]
    out["IsRepresentative"] = ["yes" if find(k) == k else "no" for k in range(n)]
    out["ClassRepresentative"] = [rep_of[k] for k in range(n)]
    out["EquivalentTo"] = [equiv_of[k] for k in range(n)]
    out["RelabellingToRepresentative"] = [gmap.get(k, "") for k in range(n)]
    out["EquivalenceNotes"] = [notes[k] for k in range(n)]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\n{name}:  {n} models  ->  {len(classes)} inequivalent configurations")
    print(f"   {sum(1 for c in classes.values() if len(c) == 1)} already unique")
    for rep, group_members in classes.items():
        if len(group_members) > 1:
            print("   " + "  =  ".join(models[k]["label"] for k in group_members))
    print(f"\nwritten to {out_path}")
    return out


# --------------------------------------------------------------------------
def classify(name: str, params, build: Callable, names: List[str],
             out_dir: str, table: Optional[Dict] = None,
             progress: int = 2000) -> List[Dict]:
    """
    params  : iterable of parameter tuples 
    build   : parameters -> 40-column basis matrix
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    mods = []
    for j, p in enumerate(params):
        b = build(*p)
        if FF.modular_invariant(b):
            mods.append((p, b))
    print(f"{name}: modular invariant parameter choices: {len(mods)} "
          f"({time.time()-t0:.0f} s)")

    reps: List[Dict] = []
    assign = []
    for j, (p, b) in enumerate(mods):
        xi = FF.additive_set(b)
        fp = FF.trace_signature(xi)
        tgt = FF.prepare_target_model(xi)
        hit = None
        for r in reps:
            if r["fp"] == fp and FF.find_equivalence(r["src"], tgt):
                hit = r["id"]
                break
        if hit is None:
            hit = len(reps)
            reps.append(dict(id=hit, fp=fp, basis=b, params=p,
                             src=FF.prepare_starting_model(xi, b), twists=list(b[3:])))
        assign.append(hit)
        if progress and (j + 1) % progress == 0:
            print(f"   {j+1}/{len(mods)} scanned, {len(reps)} classes "
                  f"({time.time()-t0:.0f} s)")

    print(f"{name}: inequivalent configurations: {len(reps)}  "
          f"({time.time()-t0:.0f} s)")

    lab = {r["id"]: f"class_{r['id']+1:02d}" for r in reps}
    if table:
        for r in reps:
            hits = [label_lookup(b, table) for (p, b), c in zip(mods, assign)
                    if c == r["id"] and label_lookup(b, table)]
            if hits:
                lab[r["id"]] = hits[0]
        matched = {v for v in lab.values() if v in table}
        print(f"{name}: table-15 entries hit: {len(matched)}/{len(table)}")
        missing = [k for k in table if k not in matched]
        if missing:
            print(f"{name}: table-15 entries not matched by any class "
                  f"representative: {', '.join(missing)}")

    return dict(reps=reps, mods=mods, assign=assign, labels=lab)


def write_classes(result, out_path, param_cols):
    """param_cols: parameters -> dict of csv columns."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    rows = []
    for r in result["reps"]:
        d = dict(param_cols(*r["params"]))
        d["Label"] = result["labels"][r["id"]]
        d["TwistBasis"] = " ; ".join(FF.twist_label(t) for t in r["twists"])
        d["NumberOfMIParameterChoices"] = result["assign"].count(r["id"])
        rows.append(d)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"written: {out_path}")


def write_all_MI(result, out_path, param_cols):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    rows = []
    for (p, _), c in zip(result["mods"], result["assign"]):
        d = dict(param_cols(*p))
        d["Class"] = result["labels"][c]
        rows.append(d)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"written: {out_path}")
