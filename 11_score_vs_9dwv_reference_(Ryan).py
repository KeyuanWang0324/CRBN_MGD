"""
Score 08's existing ternary docking models against a REAL experimental
reference structure -- PDB 9DWV (DDB1 + CRBN + PPIL4, cryo-EM, 3.5 A,
Baek/Fischer et al. 2025) -- instead of HADDOCK3's default self-consistency
pseudo-reference (the lowest-scoring model within the same run).

Why this matters: every dockq/irmsd/fnat/lrmsd number produced by 05/07/08
so far was computed against an INTERNAL reference picked from that same
candidate's own run, which makes those numbers useless for judging real
accuracy or for comparing across candidates/runs. 9DWV is a real, independent
structure showing what an actual CRBN-glue-mediated CRBN/PPIL4 ternary
complex looks like, so scoring our models against it gives a genuine
structural-plausibility measure instead.

Caveat (see chat discussion): 9DWV's bound glue (A1BC8, a triazole-thiophene
piperidine-2,6-dione) is a different chemotype than this project's own
thalidomide/phthalimide-derived analogs, and different glue caps can induce
different neosubstrate-facing surfaces. So a high dockq here should be read
as "this candidate's overall ternary architecture -- which face of CRBN,
which domain of PPIL4, roughly how the two proteins are arranged -- is
consistent with a real CRBN-glue ternary complex", NOT as "this exact
residue-level interface is correct for this candidate's specific glue."

No re-docking is done here -- this only re-evaluates the rank-1 models that
07/08 already produced (from docking_tmp/haddock3_complete_run/<candidate>/
run1/9_caprieval/), so it's fast even though it calls into HADDOCK3's own
CAPRI scoring code (haddock.libs.libcapri.CAPRI) directly, bypassing the
full run/pipeline system since we don't need to redo topology/docking.

Steps:
  1. Download PDB 9DWV as mmCIF from RCSB (plain .pdb 404s -- the structure
     is too large for the legacy format) and cache it locally.
  2. Extract just the CRBN (9DWV auth chain B) and PPIL4 (9DWV auth chain C)
     protein chains -- dropping DDB1, the bound glue, and the structural
     Zn -- and relabel them to chain A / chain B to match the convention
     every one of our own docking models uses (see 09_extract_top_structures,
     which confirmed this via `cmd.align("...chain A", "ternary and chain A")`).
  3. For each of 08's 20 candidates, resolve its rank-1 model (same TSV
     lookup logic as 09) and run HADDOCK3's own CAPRI class against the
     9DWV reference -- getting real dockq/irmsd/fnat/lrmsd/ilrmsd/global_rmsd.
  4. Write 11_reference_comparison_(Ryan).csv (one row per candidate,
     old self-consistency dockq alongside the new real-reference metrics)
     and report the Spearman correlation between the two rankings.

Run with the same environment 05/07/08 use (needs haddock3 + biopython +
scipy installed):
    python3 "11_score_vs_9dwv_reference_(Ryan).py"
"""
import csv
import gzip
import os
import shutil
import sys
import time
import urllib.request

SCRIPT_START_TIME = time.time()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_CSV = os.path.join(SCRIPT_DIR, "08_final_ternary_results_(Ryan).csv")
RUN_DIR_BASE = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_complete_run")
SCRATCH_DIR = os.path.join(SCRIPT_DIR, "docking_tmp", "11_capri_vs_9dwv_scratch")

REFERENCE_DIR = os.path.join(SCRIPT_DIR, "reference_structures")
CIF_CACHE_PATH = os.path.join(REFERENCE_DIR, "9DWV.cif")
REFERENCE_PDB_PATH = os.path.join(REFERENCE_DIR, "9DWV_reference_(Ryan).pdb")

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "11_reference_comparison_(Ryan).csv")

PDB_9DWV_CIF_URL = "https://files.rcsb.org/download/9DWV.cif"

# Confirmed via RCSB's mmCIF _atom_site.auth_asym_id: chain A = DDB1,
# chain B = CRBN, chain C = PPIL4 (RRM domain, residues ~240-318).
CRBN_SOURCE_CHAIN = "B"
PPIL4_SOURCE_CHAIN = "C"

FIELDNAMES = [
    "name", "old_dockq_selfref", "old_score",
    "dockq_vs_9dwv", "irmsd_vs_9dwv", "fnat_vs_9dwv",
    "lrmsd_vs_9dwv", "ilrmsd_vs_9dwv", "global_rmsd_vs_9dwv",
]


def download_cif():
    if os.path.exists(CIF_CACHE_PATH):
        return
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    print(f"Downloading PDB 9DWV (mmCIF) from RCSB -> {CIF_CACHE_PATH} ...")
    import ssl
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(PDB_9DWV_CIF_URL, context=ctx) as resp, open(CIF_CACHE_PATH, "wb") as out:
        shutil.copyfileobj(resp, out)
    print("Download complete.")


def build_reference_pdb():
    """Extract 9DWV's CRBN and PPIL4 protein chains only (no DDB1, no
    ligand, no Zn) and relabel them chain A (CRBN) / chain B (PPIL4) to
    match our own models' chain layout."""
    if os.path.exists(REFERENCE_PDB_PATH):
        return
    from Bio.PDB import MMCIFParser, PDBIO, Select
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("9DWV", CIF_CACHE_PATH)
    model = structure[0]

    crbn = model[CRBN_SOURCE_CHAIN]
    ppil4 = model[PPIL4_SOURCE_CHAIN]
    crbn.detach_parent()
    ppil4.detach_parent()
    crbn.id = "A"
    ppil4.id = "B"

    new_structure = Structure("9DWV_reference")
    new_model = Model(0)
    new_model.add(crbn)
    new_model.add(ppil4)
    new_structure.add(new_model)

    class ProteinOnly(Select):
        def accept_residue(self, residue):
            return residue.id[0] == " "  # drop HETATM (ligand, Zn, waters)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(REFERENCE_PDB_PATH, select=ProteinOnly())
    print(f"Wrote reference structure: {REFERENCE_PDB_PATH} "
          f"(chain A = CRBN, chain B = PPIL4, extracted from 9DWV chains "
          f"{CRBN_SOURCE_CHAIN}/{PPIL4_SOURCE_CHAIN})")


def find_top_model_path(candidate_name):
    """Same resolution logic as 09_extract_top_structures: the rank-1
    model from 08's HADDOCK3 run for this candidate."""
    caprieval_dir = os.path.join(RUN_DIR_BASE, candidate_name, "run1", "9_caprieval")
    tsv_path = os.path.join(caprieval_dir, "capri_ss.tsv")
    if not os.path.exists(tsv_path):
        return None
    with open(tsv_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    top = next((r for r in rows if r["caprieval_rank"] == "1"), None)
    if top is None:
        return None
    rel_path = top["model"]
    resolved = os.path.normpath(os.path.join(caprieval_dir, rel_path))
    if os.path.exists(resolved):
        return resolved
    if os.path.exists(resolved + ".gz"):
        return resolved + ".gz"
    return None


def decompress_if_needed(path):
    if not path.endswith(".gz"):
        return path
    out_path = os.path.join(SCRATCH_DIR, os.path.basename(path)[:-3])
    if not os.path.exists(out_path):
        with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_path


def score_candidate(candidate_name, capri_params, index):
    from pathlib import Path
    from haddock.libs.libcapri import CAPRI

    model_path = find_top_model_path(candidate_name)
    if model_path is None:
        print(f"[{candidate_name}] no rank-1 model on disk -- skipping (run 08 for this candidate first).")
        return None
    model_path = decompress_if_needed(model_path)

    capri = CAPRI(
        identificator=index,
        model=Path(model_path),
        path=Path(SCRATCH_DIR),
        reference=Path(REFERENCE_PDB_PATH),
        params=capri_params,
        ref_id=1,
        ff="aa",
    )
    result = capri.run()
    if result is None:
        print(f"[{candidate_name}] alignment against the 9DWV reference failed -- skipping.")
        return None
    return {
        "dockq_vs_9dwv": result.dockq,
        "irmsd_vs_9dwv": result.irmsd,
        "fnat_vs_9dwv": result.fnat,
        "lrmsd_vs_9dwv": result.lrmsd,
        "ilrmsd_vs_9dwv": result.ilrmsd,
        "global_rmsd_vs_9dwv": result.rmsd,
    }


def main():
    download_cif()
    build_reference_pdb()
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    if not os.path.exists(RESULTS_CSV):
        sys.exit(f"{RESULTS_CSV} not found -- run 08 first.")
    with open(RESULTS_CSV, newline="") as f:
        old_rows = list(csv.DictReader(f))

    from haddock.gear.yaml2cfg import read_from_yaml_config
    from haddock.modules.analysis.caprieval import DEFAULT_CONFIG
    capri_params = read_from_yaml_config(DEFAULT_CONFIG)

    print(f"Scoring {len(old_rows)} candidate(s) from {RESULTS_CSV} against the real 9DWV reference structure...")
    out_rows = []
    for i, row in enumerate(old_rows, 1):
        name = row["name"]
        elapsed = time.time() - SCRIPT_START_TIME
        print(f"[{i}/{len(old_rows)}] {name} (elapsed so far: {elapsed:.0f}s)")
        metrics = score_candidate(name, capri_params, i)
        out_row = {
            "name": name,
            "old_dockq_selfref": row.get("dockq", "-"),
            "old_score": row.get("score", "-"),
        }
        if metrics:
            for key in ["dockq_vs_9dwv", "irmsd_vs_9dwv", "fnat_vs_9dwv",
                        "lrmsd_vs_9dwv", "ilrmsd_vs_9dwv", "global_rmsd_vs_9dwv"]:
                out_row[key] = f"{metrics[key]:.4f}"
            print(f"    dockq_vs_9dwv={metrics['dockq_vs_9dwv']:.3f}  "
                  f"irmsd={metrics['irmsd_vs_9dwv']:.2f}  "
                  f"fnat={metrics['fnat_vs_9dwv']:.3f}  "
                  f"lrmsd={metrics['lrmsd_vs_9dwv']:.2f}")
        else:
            for key in ["dockq_vs_9dwv", "irmsd_vs_9dwv", "fnat_vs_9dwv",
                        "lrmsd_vs_9dwv", "ilrmsd_vs_9dwv", "global_rmsd_vs_9dwv"]:
                out_row[key] = "-"
        out_rows.append(out_row)

    out_rows.sort(key=lambda r: (
        r["dockq_vs_9dwv"] == "-",
        -float(r["dockq_vs_9dwv"]) if r["dockq_vs_9dwv"] != "-" else 0.0,
    ))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {OUTPUT_CSV}")

    paired = [
        (float(r["old_dockq_selfref"]), float(r["dockq_vs_9dwv"]))
        for r in out_rows
        if r["old_dockq_selfref"] != "-" and r["dockq_vs_9dwv"] != "-"
    ]
    if len(paired) >= 3:
        from scipy.stats import spearmanr
        old_vals, new_vals = zip(*paired)
        rho, p = spearmanr(old_vals, new_vals)
        print(f"\nSpearman correlation (old self-ref dockq vs new 9DWV-ref dockq), "
              f"n={len(paired)}: rho={rho:.3f}, p={p:.3f}")
        print("(A high, significant rho would suggest the old self-consistency "
              "ranking was already a decent proxy for real structural plausibility; "
              "a low/insignificant rho means it wasn't, and the 9DWV-based ranking "
              "should be trusted instead.)")

    total = time.time() - SCRIPT_START_TIME
    print(f"\nTotal script runtime: {total:.0f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
