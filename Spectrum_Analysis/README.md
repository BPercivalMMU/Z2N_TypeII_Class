# Spectrum Analysis

Computes the massless spectrum of order-two free fermionic Type II orbifold models.

### `TypeIIFreeFermioniser_v5.py`: the spectrum of a single model

Given a set of basis vectors and a GGSO phase matrix, this
computes the full massless spectrum for one model. It writes both a raw
and a processed spectrum CSV.

The script reads in the input basis and GGSOs, verifies modular
invariance and consistency, works out the surviving states sector by
sector and reports `SpectrumStats` (SUSY count, Rarita-Schwinger states,
vector/hyper multiplet counts, spin content, ...). 

Run standalone, it reads a single model's basis from
`Input_typeII/InBasis.txt` and `Input_typeII/InGSO.txt` and writes to
`Output_typeII/`. `get_model_spectra_stats_all_classes.py` (below) imports
this script's `FreeFermionModel` class to then get the spectra stats for models.

### `get_model_spectra_stats_all_classes.py`: all models in all classes

Reads every row of the 9 input CSVs in
`All_Z2N_Input_Models_updated_310826/` (one file per point-group class:
`Z2`, `Z2_2`, `Z2L`, `Z2L_2`, `Z2L_Z2R`, `Z2L_2_Z2R`, `Z2L_2_Z2R_2`,
`Z2L_Z2`, `Z2L_Z2R_Z2`), each carrying a `PaperLabel` column that gives the 
model name in terms of I,...,VI, i,...,iii, A,...,E, a,...,c notation of paper 
(see below). For each row it builds the corresponding basis vector matrix, 
scans over the IIA/IIB choice and that class's GGSO phase
variant(s) that impose key projections determining the twisted sectors 
and calls into `TypeIIFreeFermioniser_v5.py` to compute the processed spectrum 
for each combination.

A few key details:
- **Basis construction**: each point group's basis vectors are built
  from a row's tuple-valued shift parameters (e.g. `n`, `N` for $B_1$, etc) by
  the `build_basis_*` functions, one per class.
- **GGSO scanning**: every class always scans both IIA and IIB; classes
  with more than one independent twist also scan the structural GGSO
  phases between twists and (optionally) SUSY-breaking phases.
- **Outputs**:
  - `Processed_Spectra_SUSY/{class}/{IIA,IIB}/{run_label}_processed.csv`
   gives processed spectra (SUSY-preserving runs) for each model
  - `Processed_Spectra_non_SUSY/{class}/{IIA,IIB}/...` the same, for
    SUSY-breaking variants (not enabled here as focused on preserved SUSY)

### `build_nonsusy_enhanced_models.py`

A small additional script that reuses the same machinery to build and run the two
specific non-SUSY enhanced models given in the paper: `Z2L_2_Z2R` with N=0->1 and 
`Z2L_2_Z2R_2` with N=0->2. Inputs are in `inputs_nonSUSY_enhanced/` and output:
`Non_SUSY_enhancement_models/`.

### `Input_typeII/supermultiplets.csv`

The massless multiplets of each extended supersymmetry, taken from table 20 of
the paper, as spin-state counts `[n0, n_half, n1, n_3half, n2]`. Both the
`_processed` files and the summary stats read their multiplets from here, so
the supermultiplet matching quoted in the two places always agrees. If a
SUSY level is missing from this file the code falls back to computing the
multiplets itself.

## Input model labelling

Every row in `All_Z2N_Input_Models_updated_310826/` carries a
`PaperLabel` column giving that model's twist-vector classification in
the paper's convention: `I`,..,`VI` (e.g. Z2L^2: $b_1,b_2$), `i`,...,`iii`
(single $b_1$/$b_{\bar1}$), `a`–`c` (symmetric $b_{1\bar1}$), and
`A`,...,`E` (symmetric $b_{2\bar2}$), composed with `-` for classes built
from more than one twist (e.g. `IV-ii.2`, `iii-iii-E.4`).
