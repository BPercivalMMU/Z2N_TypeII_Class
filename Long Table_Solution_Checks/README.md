# Long Table Solution Checks

Classification, from scratch, of the order-two T-fold point groups on the
SO(12) lattice at the free fermionic point. `Z2L_2_Z2R`, `Z2L_2_Z2R_2`,
`Z2L_Z2` and `Z2L_Z2R_Z2` each have their own subfolder with a
`classify_*.py` script; the other point-group subfolders (`Z2L_2`,
`Z2R_Z2L`) are unrelated legacy checks and not covered here.

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
   `FF_equivalence_checker_master.py`, or the point-group-specific variant
   copied into a subfolder when the search needs a specialised layout.

`Z2L_2_Z2R` and `Z2L_Z2` route through the shared `classification_tools_all.py`
driver and `pointgroup_specs.py` (which defines each point group's
parameters and basis-vector builder). `Z2L_2_Z2R_2` and `Z2L_Z2R_Z2` are
larger scans structured as one-sided-data x one-sided-data and call their own
local `FF_equivalence_checker_*.py` directly instead.

## Shared modules (this folder)

- `classification_tools_all.py` — enumerate/reduce/report driver used by the
  `Z2L_2_Z2R` and `Z2L_Z2` scripts.
- `FF_equivalence_checker_master.py` — point-group-independent equivalence
  search (`find_equivalence`) and modular-invariance checks.
- `pointgroup_specs.py` — per-point-group parameter ranges and basis-vector
  builders, keyed by name in `SPECS`.

## Running a script

```
python Z2L_2_Z2R/classify_Z2L_2_Z2R.py [--out-dir DIR] [--no-all-mi]
python Z2L_Z2/classify_Z2L_Z2.py [--out-dir DIR] [--no-all-mi]
python Z2L_2_Z2R_2/classify_Z2L_2_Z2R_2.py [--out-dir DIR] [--reference FILE] [--limit-left N] [--no-all-mi]
python Z2L_Z2R_Z2/classify_Z2L_Z2R_Z2.py [--out-dir DIR] [--reference FILE]
```

`--reference` (where supported) takes a csv of known models with a
`PaperLabel` column, used to name the inequivalent classes found.
`--limit-left` restricts the `Z2L_2_Z2R_2` scan to the first N left-hand data,
for a quick smoke test.

## Outputs

Each subfolder's `Outputs/` directory holds:

- `*_inequivalent.csv` — one row per inequivalent configuration found.
- `*_all_MI.csv` — every modular-invariant parameter choice, labelled by the
  equivalence class it belongs to.
- `*_equivalence_check.csv` — diagnostics from the equivalence-reduction step.