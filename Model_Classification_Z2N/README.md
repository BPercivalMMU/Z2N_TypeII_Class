# Z2N Model Classification Table

Classification, from scratch, of the order-two T-fold point groups on the
SO(12) lattice at the free fermionic point. `Z2L_2`, `Z2L_Z2R`, `Z2L_2_Z2R`,
`Z2L_2_Z2R_2`, `Z2L_Z2` and `Z2L_Z2R_Z2` each have their own subfolder with a
`classify_*.py` script. The other, simpler point-groups models are solved analytically and given in the paper.

## What each script does

1. **Enumerate.** Scan the full range of the parameters defining that point
   group's twist basis vectors and keep every choice for which the basis
   satisfies the modular invariance conditions imposed directly on
   the vectors. 
2. **Reduce to inequivalent classes.** Survivors are quotiented by the
   equivalence relations E1-E3 of section 4.1: 
   E1: GL(|B|;Z) basis changes,
   E2: `y^i <-> w^i`, and
   E3: holomorphic/anti-holomorphic index permutations
   with an exhaustive `G_L x G_R` search: `find_equivalence()` in
   `FF_equivalence_checker_master.py`, shared by all six scripts.

`Z2L_2`, `Z2L_Z2R`, `Z2L_2_Z2R` and `Z2L_Z2` use `classification_tools_all.py`  and `pointgroup_specs.py` 
(which defines each point group's parameters and basis builder).
`Z2L_2_Z2R_2` and `Z2L_Z2R_Z2` use `FF_equivalence_checker_master.py` directly
rather than through `classify()`. 

## Shared modules (this folder)

- `classification_tools_all.py` — used by the
  `Z2L_2`, `Z2L_Z2R`, `Z2L_2_Z2R` and `Z2L_Z2` scripts.
- `FF_equivalence_checker_master.py` — equivalence and modular-invariance checks.
- `pointgroup_specs.py` — per-point group parameter ranges, basis
  builders and table-15 labels.

## Running a script

```
python Z2L_2/classify_Z2L_2.py [--out-dir DIR] [--no-all-mi]
python Z2L_Z2R/classify_Z2L_Z2R.py [--out-dir DIR] [--no-all-mi]
python Z2L_2_Z2R/classify_Z2L_2_Z2R.py [--out-dir DIR] [--no-all-mi]
python Z2L_Z2/classify_Z2L_Z2.py [--out-dir DIR] [--no-all-mi]
python Z2L_2_Z2R_2/classify_Z2L_2_Z2R_2.py [--out-dir DIR] [--limit-left N] [--no-all-mi]
python Z2L_Z2R_Z2/classify_Z2L_Z2R_Z2.py [--out-dir DIR]
```

`--limit-left` allows for th `Z2L_2_Z2R_2` to be scanned initially for equivalences on the 
left twist basis vectors to make more efficient.

## Outputs

Each subfolder's `Outputs/` directory holds:

- `*_inequivalent.csv` — one row per inequivalent configuration found.
- `*_all_MI.csv` — every modular-invariant parameter choice, labelled by the
  equivalence class it belongs to.
