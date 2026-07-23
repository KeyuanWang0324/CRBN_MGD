"""
Extract the representative 3D structure (the rank-1, best-scoring model of
the top-ranked cluster) for each of 08's top-ranked candidates, so they can
be opened directly in PyMOL for visual inspection/comparison.

For a given candidate, 08_haddock3_ternary_complete_(Ryan).py's HADDOCK3
run writes docking_tmp/haddock3_complete_run/<candidate>/run1/9_caprieval/
capri_ss.tsv, which lists every sampled model with its caprieval_rank; the
row with caprieval_rank == 1 names the single best-scoring model (e.g.
"../6_emref/emref_3.pdb"), which HADDOCK3 gzips in place once the run
finishes. This script resolves that path, decompresses it, and copies it
to ternary_structures/ with a clear name.

IMPORTANT: that extracted structure has NO drug molecule in it. 07/08's
HADDOCK3 config only ever docks CRBN_RECEPTOR_ONLY_PDB against PPIL4_PDB --
the candidate's own atoms are used upstream (in 06's Vina docking) only to
derive which CRBN residues to restrain against, then discarded. To also
show the actual ligand, this script uses PyMOL (headless, via PYMOL_BIN) to
align 06's CRBN+ligand complex (docking_tmp/haddock3_novel_candidate/
<candidate>/CRBN_candidate_complex.pdb) onto the ternary structure's CRBN
chain, then merges in just the ligand atoms -- giving a combined
CRBN+ligand+PPIL4 file (09_best_model_with_ligand_<candidate>_(Ryan).pdb).
If PyMOL isn't found, the ligand-free file is still written; only the
ligand-merge step is skipped.

Run with any Python that can read 08_final_ternary_results_(Ryan).csv
(no haddock3/rdkit/vina dependency needed):
    python3 "09_extract_top_structures_(Ryan).py"
"""
import csv
import gzip
import os
import shutil
import subprocess
import sys
import time

SCRIPT_START_TIME = time.time()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_CSV = os.path.join(SCRIPT_DIR, "08_final_ternary_results_(Ryan).csv")
RUN_DIR_BASE = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_complete_run")
VINA_LIGAND_COMPLEX_DIR = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_novel_candidate")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ternary_structures")

# How many of 08's top-ranked (by dockq) candidates to extract a structure
# for. RESULTS_CSV is already sorted best-dockq-first.
TOP_N = 5

# PyMOL is only used for the optional ligand-merge step (aligning the drug
# molecule from 06's output onto the ligand-free ternary structure) -- not
# required for the base extraction. Add other install locations here if
# needed; the plain "pymol" entry covers Linux/most package managers.
PYMOL_CANDIDATES = ["/Applications/PyMOL.app/Contents/bin/pymol", "pymol"]

_MERGE_LIGAND_SCRIPT = """
import sys
from pymol import cmd

ternary_path, complex_path, out_path = sys.argv[-3:]
cmd.load(ternary_path, "ternary")
cmd.load(complex_path, "ligcomplex")
rms = cmd.align("ligcomplex and polymer and chain A", "ternary and chain A")
print(f"ALIGN_RMSD {rms[0]:.3f} {rms[1]}")
cmd.create("combined", "(ternary) or (ligcomplex and resn LIG)")
n_lig = cmd.count_atoms("combined and resn LIG")
cmd.save(out_path, "combined")
print(f"LIG_ATOMS {n_lig}")
"""


def find_pymol_bin():
    for candidate in PYMOL_CANDIDATES:
        if os.path.isabs(candidate):
            if os.path.exists(candidate):
                return candidate
        elif shutil.which(candidate):
            return candidate
    return None


def find_top_model_path(candidate_name):
    """Return the absolute path to candidate_name's rank-1 model file
    (decompressed if needed to a .gz sibling), or None if not found."""
    caprieval_dir = os.path.join(RUN_DIR_BASE, candidate_name, "run1", "9_caprieval")
    tsv_path = os.path.join(caprieval_dir, "capri_ss.tsv")
    if not os.path.exists(tsv_path):
        return None

    with open(tsv_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    top = next((r for r in rows if r["caprieval_rank"] == "1"), None)
    if top is None:
        return None

    rel_path = top["model"]  # e.g. "../6_emref/emref_3.pdb"
    resolved = os.path.normpath(os.path.join(caprieval_dir, rel_path))
    if os.path.exists(resolved):
        return resolved
    if os.path.exists(resolved + ".gz"):
        return resolved + ".gz"
    return None


def extract_structure(candidate_name):
    """Decompress (if needed) candidate_name's rank-1 model into OUTPUT_DIR.
    Returns the written path, or None if the model couldn't be found."""
    src_path = find_top_model_path(candidate_name)
    if src_path is None:
        print(f"[{candidate_name}] no rank-1 model found (has 08 been run for this candidate?) -- skipping.")
        return None

    out_path = os.path.join(OUTPUT_DIR, f"09_best_model_{candidate_name}_(Ryan).pdb")
    if src_path.endswith(".gz"):
        with gzip.open(src_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copyfile(src_path, out_path)
    print(f"[{candidate_name}] wrote {out_path} (from {os.path.relpath(src_path, SCRIPT_DIR)})")
    return out_path


def merge_ligand(candidate_name, ternary_path, pymol_bin, script_path):
    """Align 06's CRBN+ligand complex onto the ternary structure's CRBN
    chain and save a combined CRBN+ligand+PPIL4 PDB. Returns the written
    path, or None if 06's complex file for this candidate doesn't exist."""
    complex_path = os.path.join(VINA_LIGAND_COMPLEX_DIR, candidate_name, "CRBN_candidate_complex.pdb")
    if not os.path.exists(complex_path):
        print(f"[{candidate_name}] {complex_path} not found -- run 06 for this candidate first. "
              "Skipping ligand merge.")
        return None

    out_path = os.path.join(OUTPUT_DIR, f"09_best_model_with_ligand_{candidate_name}_(Ryan).pdb")
    result = subprocess.run(
        [pymol_bin, "-cq", script_path, "--", ternary_path, complex_path, out_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(out_path):
        print(f"[{candidate_name}] PyMOL ligand-merge failed:\n{result.stdout}\n{result.stderr}")
        return None

    rmsd_line = next((l for l in result.stdout.splitlines() if l.startswith("ALIGN_RMSD")), "")
    lig_line = next((l for l in result.stdout.splitlines() if l.startswith("LIG_ATOMS")), "")
    print(f"[{candidate_name}] wrote {out_path} ({rmsd_line}, {lig_line})")
    return out_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(RESULTS_CSV):
        sys.exit(f"{RESULTS_CSV} not found -- run 08 first.")
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = [r["name"] for r in rows[:TOP_N]]
    print(f"Extracting top structures for the top {len(candidates)} candidate(s) from {RESULTS_CSV}: "
          f"{', '.join(candidates)}")

    pymol_bin = find_pymol_bin()
    script_path = None
    if pymol_bin:
        script_path = os.path.join(OUTPUT_DIR, "_pymol_merge_ligand_script.py")
        with open(script_path, "w") as f:
            f.write(_MERGE_LIGAND_SCRIPT)
    else:
        print("PyMOL not found (checked: " + ", ".join(PYMOL_CANDIDATES) + ") -- "
              "will still write the ligand-free structures, but skipping the ligand-merge step.")

    written = []
    for i, candidate_name in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {candidate_name}")
        out_path = extract_structure(candidate_name)
        if out_path:
            written.append(out_path)
            if pymol_bin:
                merged_path = merge_ligand(candidate_name, out_path, pymol_bin, script_path)
                if merged_path:
                    written.append(merged_path)

    if script_path and os.path.exists(script_path):
        os.remove(script_path)

    print(f"\nWrote {len(written)} file(s) to {OUTPUT_DIR}")

    total = time.time() - SCRIPT_START_TIME
    print(f"Total script runtime: {total:.0f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
