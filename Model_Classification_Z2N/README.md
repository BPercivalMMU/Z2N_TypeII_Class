# Z2N Model Classification Table

Classification, from scratch, of the order-two T-fold point groups on the
SO(12) lattice at the free fermionic point. `Z2L_2`, `Z2L_Z2R`, `Z2L_2_Z2R`,
`Z2L_2_Z2R_2`, `Z2L_Z2` and `Z2L_Z2R_Z2` each have their own subfolder with a
`classify_*.py` script. The other point-groups models are solved analytically and given in the paper.

## What each script does

1. **Enumerate.** Scan the full range of the parameters defining that point
   group's twist basis vectors and keep every choice for which the basis
   satisfies the modular invariance conditions (3.4) — imposed directly on
   the vectors, including the two-loop four-fold overlap condition. No
   hand-derived parameter relation, no standard-form restriction (A1-A4,
   S1-S2 of section 4.3), no `|N| <= 3` cut and no gauge fixing enters.
2. **Reduce to inequivalent classes.** Survivors are quotiented by the
   equivalence relations E1-E3 of section 4.1 (GL(|B|;Z) basis changes,
   `y^i <-> w^i`, and holomorphic/anti-holomorphic index permutations) with
   an exhaustive `G_L x G_R` search — `find_equivalence()` in
   `FF_equivalence_checker_master.py`, shared by all six scripts.

`Z2L_2`, `Z2L_Z2R`, `Z2L_2_Z2R` and `Z2L_Z2` route through the shared
`classification_tools_all.py` driver and `pointgroup_specs.py` (which defines
each point group's parameters and basis-vector builder, and — where the paper
gives one — a `table` of table-15 labels used to name the classes found).
`Z2L_2_Z2R_2` and `Z2L_Z2R_Z2` are larger scans structured as one-sided-data
x one-sided-data, so they drive `FF_equivalence_checker_master.py` directly
rather than through `classify()`, and their classes are left labelled
`class_01`, `class_02`, ... since no `table` is set up for them yet.

## Shared modules (this folder)

- `classification_tools_all.py` — enumerate/reduce/report driver used by the
  `Z2L_2`, `Z2L_Z2R`, `Z2L_2_Z2R` and `Z2L_Z2` scripts.
- `FF_equivalence_checker_master.py` — point-group-independent equivalence
  search (`find_equivalence`) and modular-invariance checks.
- `pointgroup_specs.py` — per-point-group parameter ranges, basis-vector
  builders and table-15 labels, keyed by name in `SPECS`.

## Running a script

```
python Z2L_2/classify_Z2L_2.py [--out-dir DIR] [--no-all-mi]
python Z2L_Z2R/classify_Z2L_Z2R.py [--out-dir DIR] [--no-all-mi]
python Z2L_2_Z2R/classify_Z2L_2_Z2R.py [--out-dir DIR] [--no-all-mi]
python Z2L_Z2/classify_Z2L_Z2.py [--out-dir DIR] [--no-all-mi]
python Z2L_2_Z2R_2/classify_Z2L_2_Z2R_2.py [--out-dir DIR] [--limit-left N] [--no-all-mi]
python Z2L_Z2R_Z2/classify_Z2L_Z2R_Z2.py [--out-dir DIR]
```

`--limit-left` restricts the `Z2L_2_Z2R_2` scan to the first N left-hand data,
for a quick smoke test.

## Outputs

Each subfolder's `Outputs/` directory holds:

- `*_inequivalent.csv` — one row per inequivalent configuration found.
- `*_all_MI.csv` — every modular-invariant parameter choice, labelled by the
  equivalence class it belongs to.
