"""Build input files and run spectra for the two non-SUSY-enhancement models.

Model 1 (N=0->1): Z2L_2_Z2R  — 6 basis vectors {1,S,Sbar,B1,B2,B1b}
Model 2 (N=0->2): Z2L_2_Z2R_2 — 7 basis vectors {1,S,Sbar,B1,B2,B1b,B2b}

Both use the IIB GGSO template with the following overrides:
  C(Sbar,Bα) = +1  (Bα = B1, B2)
  C(S, B̄β)  = +1  (B̄β = B1b [model 1] or B1b, B2b [model 2])
  C(Bα, B̄β) = +1  (every left/right pair; already the template's default)
  C(S, Bα)   = -1  (default in template)
  C(Sbar, B̄β) = -1 (default in template)

Inputs are saved to:  inputs_nonSUSY_enhanced/
Outputs are saved to: Non_SUSY_enhancement_models/
"""

import os
import sys
import importlib.util
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

STATS_SCRIPT   = os.path.join(_HERE, "get_model_spectra_stats_all_classes.py")
FF_SCRIPT      = os.path.join(_HERE, "TypeIIFreeFermioniser_v5.py")
MULT_CSV       = os.path.join(_HERE, "Input_typeII", "supermultiplets.csv")

INPUT_DIR  = os.path.join(_HERE, "inputs_nonSUSY_enhanced")
OUTPUT_DIR = os.path.join(_HERE, "Non_SUSY_enhancement_models")

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Import helpers from the stats script ──────────────────────────────────────

def _import_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # must be in sys.modules before exec so dataclasses resolve __module__
    spec.loader.exec_module(mod)
    return mod

stats_mod = _import_module(STATS_SCRIPT, "stats_mod")
ff_mod    = _import_module(FF_SCRIPT,    "ff_mod")

gso_from_template   = stats_mod.gso_from_template
_template_7x7       = stats_mod._template_7x7
FreeFermionModel    = ff_mod.FreeFermionModel

# ── Basis vectors (40-entry pure-twist form) ───────────────────────────────────

#           psi12,chi123456    y123456     w123456      psib12,chibar123456   ybar123456   wbar123456
_1    = [1,1,1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1]
_S    = [1,1,1,1,1,1,1,1, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0]
_Sbar = [0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0, 0,0,0,0,0,0]
_B1   = [0,0,0,0,1,1,1,1, 0,0,1,1,1,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0]
_B2   = [0,0,1,1,0,0,1,1, 1,1,0,0,1,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0]
_B1b  = [0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,1,1,1,1, 0,0,1,1,1,1, 0,0,0,0,0,0]
_B2b  = [0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,1,1,0,0,1,1, 1,1,0,0,1,1, 0,0,0,0,0,0]

BASIS_M1 = np.array([_1, _S, _Sbar, _B1, _B2, _B1b], dtype=int)   # shape (6, 40)
BASIS_M2 = np.array([_1, _S, _Sbar, _B1, _B2, _B1b, _B2b], dtype=int)  # shape (7, 40)

# ── GGSO overrides (indices into per-model submatrix) ─────────────────────────
# Row/col mapping for both models:
#   0='1', 1=S, 2=Sbar, 3=B1, 4=B2, 5=B1b, [6=B2b model 2 only]
#
# SUSY-projection phases (non-default vs IIB template):
#   C(Sbar,B1) = +1 → (2,3): projects left gravitini from S sector via B1
#   C(Sbar,B2) = +1 → (2,4): projects left gravitini from S sector via B2
#   C(S,B1b)   = +1 → (1,5): projects right gravitini from Sbar sector via B1b
#   [m2] C(S,B2b) = +1 → (1,6): projects right gravitini via B2b
#
# RS-enabling phases:
#   RS spin-3/2 requires psib12 to survive in the S+B1+B2 sector.
#   That sector's B1b projector has exponent=0 (B1b is purely right-compact),
#   so psib12 passes iff C(B1b, S+B1+B2) = delta_sec = -1, i.e.
#   C(B1b,S)*C(B1b,B1)*C(B1b,B2) = -1.
#   With C(B1b,S)=C(B1b,B1)=+1 fixed, we need C(B2,B1b) = -1 → (4,5):-1.
#   This also enables RS_L in the S+B2 sector.
#   For model 2: B2b projector imposes the same constraint on S+B1+B2, so
#   additionally C(B1,B2b) = -1 → (3,6):-1 is required.  This is symmetric
#   and also enables RS_R in the Sbar+B1b+B2b sector (B3b supersector).
#   The S-sector gravitino remains killed by B1 (exponent=0, C(B1,S)=+1,
#   delta=-1 → rhs=-1 ≠ +1 = lhs) regardless of the RS-enabling phases.

OVERRIDES_M1 = {(2, 3): +1, (2, 4): +1, (1, 5): +1, (4, 5): -1}
OVERRIDES_M2 = {(2, 3): +1, (2, 4): +1, (1, 5): +1, (1, 6): +1,
                (4, 5): -1, (3, 6): -1}

# ── Build GSO matrices ────────────────────────────────────────────────────────

GSO_M1 = gso_from_template(BASIS_M1, "Z2L_2_Z2R",   "IIB", upper_overrides=OVERRIDES_M1)
GSO_M2 = gso_from_template(BASIS_M2, "Z2L_2_Z2R_2", "IIB", upper_overrides=OVERRIDES_M2)

# ── Save input txt files ──────────────────────────────────────────────────────

def save_txt(arr: np.ndarray, path: str) -> None:
    np.savetxt(path, arr, fmt="%d", delimiter=" ")
    print(f"  saved: {path}")

print("\n[build] Saving input files ...")
save_txt(BASIS_M1, os.path.join(INPUT_DIR, "basis_N01_Z2L_2_Z2R.txt"))
save_txt(GSO_M1,   os.path.join(INPUT_DIR, "gso_N01_Z2L_2_Z2R.txt"))
save_txt(BASIS_M2, os.path.join(INPUT_DIR, "basis_N02_Z2L_2_Z2R_2.txt"))
save_txt(GSO_M2,   os.path.join(INPUT_DIR, "gso_N02_Z2L_2_Z2R_2.txt"))

# ── Print GSO matrices for verification ───────────────────────────────────────

def print_gso(label: str, gso: np.ndarray, names: list) -> None:
    print(f"\n{label} GGSO matrix:")
    w = max(len(n) for n in names)
    print("  " + " " * w + "  " + "  ".join(f"{n:>{w}}" for n in names))
    for i, row_name in enumerate(names):
        vals = "  ".join(f"{gso[i,j]:>{w}}" for j in range(len(names)))
        print(f"  {row_name:{w}}  {vals}")

NAMES_M1 = ["1", "S", "Sbar", "B1", "B2", "B1b"]
NAMES_M2 = ["1", "S", "Sbar", "B1", "B2", "B1b", "B2b"]
print_gso("Model 1 (N=0->1)", GSO_M1, NAMES_M1)
print_gso("Model 2 (N=0->2)", GSO_M2, NAMES_M2)

# ── Run FreeFermioniser ───────────────────────────────────────────────────────

MODELS = [
    ("N01_Z2L_2_Z2R",   BASIS_M1, GSO_M1),
    ("N02_Z2L_2_Z2R_2", BASIS_M2, GSO_M2),
]

print("\n[run] Computing spectra ...")
for label, basis, gso in MODELS:
    print(f"\n  === {label} ===")
    ff = FreeFermionModel.from_arrays(
        basis=basis, gso=gso, label=label, type_ii="IIB",
        multiplet_csv=MULT_CSV,
    )
    df_raw, df_proc, stats = ff.compute()

    out_csv = os.path.join(OUTPUT_DIR, f"{label}_processed.csv")
    ff.write_processed_csv(
        df_proc, out_csv,
        stats.twisted_supersectors,
        stats.rs_supersectors,
        stats.vh_t_supersectors,
    )
    print(f"  MI_OK={stats.mi_ok}  N_SUSY_L={stats.n_susy_L}  N_SUSY_R={stats.n_susy_R}")
    print(f"  Saved: {out_csv}")

print("\n[done]")
