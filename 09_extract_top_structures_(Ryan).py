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

Run with any Python that can read 08_final_ternary_results_(Ryan).csv
(no haddock3/rdkit/vina dependency needed):
    python3 "09_extract_top_structures_(Ryan).py"
"""
import csv
import gzip
import os
import shutil
import sys
import time

SCRIPT_START_TIME = time.time()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_CSV = os.path.join(SCRIPT_DIR, "08_final_ternary_results_(Ryan).csv")
RUN_DIR_BASE = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_complete_run")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ternary_structures")

# How many of 08's top-ranked (by dockq) candidates to extract a structure
# for. RESULTS_CSV is already sorted best-dockq-first.
TOP_N = 5


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(RESULTS_CSV):
        sys.exit(f"{RESULTS_CSV} not found -- run 08 first.")
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = [r["name"] for r in rows[:TOP_N]]
    print(f"Extracting top structures for the top {len(candidates)} candidate(s) from {RESULTS_CSV}: "
          f"{', '.join(candidates)}")

    written = []
    for i, candidate_name in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {candidate_name}")
        out_path = extract_structure(candidate_name)
        if out_path:
            written.append(out_path)

    print(f"\nWrote {len(written)}/{len(candidates)} structure(s) to {OUTPUT_DIR}")

    total = time.time() - SCRIPT_START_TIME
    print(f"Total script runtime: {total:.0f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
