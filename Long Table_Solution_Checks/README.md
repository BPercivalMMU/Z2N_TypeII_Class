# Long Table Solution Checks

Each subfolder corresponds to one order-two point group (twist/shift
combination) and contains a `Checker_*.py` script that finds the
**inequivalent** boundary-condition solutions for that point group and
cross-checks them against the hand-derived solutions used in the paper.

## What each script does

1. **Enumerate.** Using the [Z3](https://github.com/Z3Prover/z3) SMT
   solver, declare the point group's shift/twist parameters as 0/1
   variables and add the modular-invariance constraints those parameters
   must satisfy (plus the domain restrictions on which combinations are
   allowed, e.g. `n12 ∈ {(0,0),(1,0),(1,1)}`). Repeatedly solve and block
   the found model to enumerate *every* raw solution.
2. **Quotient to unique representatives.** Raw solutions are reduced to
   inequivalent classes using an equivalence key built from
   rotation/reflection-invariant quantities (e.g. squared lengths
   `|N|², |M|², |N-M|²`). One representative per equivalence class is
   kept.
3. **Check the hand solutions.** The paper's hand-derived solutions
   (`*_stefan_solutions.csv`) are read in, verified against the same
   constraints, and mapped onto the equivalence classes found in step 2 —
   flagging any hand solution that fails the constraints, any duplicate
   mappings, and any equivalence classes the hand solutions never hit.

## Outputs

- `*_unique_solutions.csv` — the full set of inequivalent solutions found
  by the solver.
- `*_stefan_mapped.csv` — each hand solution, annotated with whether it
  satisfies all constraints and which equivalence-class index (if any)
  it maps to.

The console output from each script also reports repeated indices,
missing (unmatched) equivalence classes, and any unmatched hand solutions.
