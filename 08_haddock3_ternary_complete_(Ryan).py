"""
COMPLETE HADDOCK3 ternary-docking routine (full sampling + emref + clustfcc)
for a single CRBN-glue candidate vs PPIL4.

06_dock_candidate_crbn_(Ryan).py and 07_haddock3_ternary_novel_candidate_(Ryan).py
run a cheap, truncated HADDOCK3 config (rigidbody sampling=20, flexref on the
top 10, no emref, no clustering) to rank many candidates quickly. This script
runs the COMPLETE stock HADDOCK3 protein-protein routine instead -- rigidbody
sampling=1000, flexref + emref refinement on the top 200, then clustfcc
clustering -- but only on ONE candidate: the finalist you pick after looking
at 07's comparison table.

This is far more expensive than 06/07 (rough estimate: 45 min - 1.5 hr on an
18-core machine, vs. ~3 min for 07's lite pass per candidate) -- run it on
the finalist only, not as a screen.

Run in the haddock3 venv:
    source .venv-haddock3/bin/activate
    python3 "08_haddock3_ternary_complete_(Ryan).py"
"""
import csv
import glob
import os
import shutil
import subprocess
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# This script needs the haddock3/pdb-tools CLIs (haddock3, haddock3-restraints,
# pdb_chain), which live in .venv-haddock3/bin. If that's not on PATH --
# e.g. the venv wasn't activated, or the IDE's Run button used a different
# interpreter -- relaunch under it automatically instead of failing deep
# inside a subprocess call.
if shutil.which("haddock3") is None:
    haddock_venv_bin = os.path.join(SCRIPT_DIR, ".venv-haddock3", "bin")
    haddock_python = os.path.join(haddock_venv_bin, "python3")
    env = os.environ.copy()
    env["PATH"] = haddock_venv_bin + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = os.path.join(SCRIPT_DIR, ".venv-haddock3")
    os.execve(haddock_python, [haddock_python] + sys.argv, env)

from Bio import Align
from Bio.Align import substitution_matrices

VINA_OUT_DIR = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_novel_candidate")
SCREENING_SUMMARY_CSV = os.path.join(VINA_OUT_DIR, "screening_summary.csv")

# Set this to the winning candidate's name from 07's final comparison table
# (best dockq). Leave blank to auto-pick the best combined_affinity candidate
# from 06's screening summary instead.
CANDIDATE_NAME = ""

# CNS's "@@" include syntax truncates paths at "(" -- keep this filename
# parenthesis-free since it's fed directly to HADDOCK3 as a molecule.
CRBN_RECEPTOR_ONLY_PDB = os.path.join(SCRIPT_DIR, "CRBN_receptor_thalidomide_Ryan.pdb")
PPIL4_SOURCE_PDB = os.path.join(SCRIPT_DIR, "PPIL4_alphafold_(Ryan).pdb")
RUN_DIR_BASE = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_complete_run")
PPIL4_PDB = os.path.join(RUN_DIR_BASE, "PPIL4_chainB.pdb")

# HADDOCK3's own `ncores` default is 4 regardless of machine size -- bump it
# to (all cores - 1) so CNS jobs actually use the available hardware.
NCORES = max(1, (os.cpu_count() or 4) - 1)

# (step index, module name, per-model count if it writes one *_N.pdb.gz file
# per model else None, rough share of total wall-clock time) -- mirrors the
# [modules] laid out in main()'s cfg below. Weights are from the estimate in
# the 06/07-vs-08 comparison (rigidbody/flexref/emref dominate; the caprieval/
# seletop/clustfcc/topoaa stages are comparatively instant) and are only used
# to turn "which step is running" into one rough live percentage -- not a
# precise timing model.
STEP_PLAN = [
    (0, "topoaa", None, 0.005),
    (1, "rigidbody", 1000, 0.30),
    (2, "caprieval", None, 0.04),
    (3, "seletop", None, 0.005),
    (4, "flexref", 200, 0.30),
    (5, "caprieval", None, 0.02),
    (6, "emref", 200, 0.26),
    (7, "caprieval", None, 0.02),
    (8, "clustfcc", None, 0.035),
    (9, "caprieval", None, 0.015),
]


def estimate_progress(run_dir, step_plan):
    """Best-effort (step, fraction-within-step, overall %) from what's on disk
    so far. Per-model steps (rigidbody/flexref/emref) count actual completed
    *_N.pdb[.gz] files against the known model count for an exact within-step
    fraction -- models are written as plain .pdb while the stage is running
    and only gzipped to .pdb.gz once the whole stage finishes, so both must
    be matched or in-progress stages read as 0%. Other steps just count as
    done once the next step's directory appears (they're quick, so this is a
    small share of the total anyway)."""
    completed_weight = 0.0
    current = None
    for idx, name, expected, weight in step_plan:
        step_dir = os.path.join(run_dir, f"{idx}_{name}")
        if not os.path.isdir(step_dir):
            break

        next_dir = (os.path.join(run_dir, f"{idx + 1}_{step_plan[idx + 1][1]}")
                    if idx + 1 < len(step_plan) else None)
        if next_dir and os.path.isdir(next_dir):
            completed_weight += weight
            continue

        if expected:
            count = len({os.path.basename(p).split(".")[0]
                         for p in glob.glob(os.path.join(step_dir, f"{name}_*.pdb*"))})
            frac = min(count / expected, 1.0)
            current = (idx, name, count, expected)
        else:
            frac = 0.5
            current = (idx, name, None, None)
        completed_weight += weight * frac
        break

    return completed_weight, current


def ppil4_pocket_residues():
    """Same CypA-homology active-site mapping used throughout this project."""
    cypa = ("MVNPTVFFDIAVDGEPLGRVSFELFADKVPKTAENFRALSTGEKGFGYKGSCFHRIIPGF"
            "MCQGGDFTRHNGTGGKSIYGEKFEDENFILKHTGPGILSMANAGPNTNGSQFFICTAKTE"
            "WLDGKHVVFGKVKEGMNIVEAMERFGSRNGKTSKKITIADCGQLE")
    ppil4_full = ("MAVLLETTLGDVVIDLYTEERPRACLNFLKLCKIKYYNYCLIHNVQRDFIIQTGDPTGTGRGGESIFGQLYGDQASFF"
                  "EAEKVPRIKHKKKGTVSMVNNGSDQHGSQFLITTGENLDYLDGVHTVFGEVTEGMDIIKKINETFVDKDFVPYQDIRI"
                  "NHTVILDDPFDDPPDLLIPDRSPEPTREQLDSGRIGADEEIDDFKGRSAEEVEEIKAEKEAKTQAILLEMVGDLPDAD"
                  "IKPPENVLFVCKLNPVTTDEDLEIIFSRFGPIRSCEVIRDWKTGESLCYAFIEFEKEEDCEKAFFKMDNVLIDDRRIH"
                  "VDFSQSVAKVKWKGKGGKYTKSDFKEYEKEQDKPPNLVLKDKVKPKQDTKYDLILDEQAEDSKSSHSHTSKKHKKKTH"
                  "HCSEEKEDEDYMPIKNTNQDIYREMGFGHYEEEESCWEKQKSEKRDRTQNRSRSRSRERDGHYSNSHKSKYQTDLYER"
                  "ERSKKRDRSRSPKKSKDKEKSKYR")
    ppil4_domain = ppil4_full[:180]

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "global"
    aln = aligner.align(cypa, ppil4_domain)[0]

    active_site_cypa = [55, 60, 61, 63, 72, 101, 102, 103, 111, 113, 121, 122, 126]
    aligned_cypa, aligned_ppil4 = aln[0], aln[1]
    cypa_pos = ppil4_pos = 0
    mapping = {}
    for c, p in zip(aligned_cypa, aligned_ppil4):
        if c != "-":
            cypa_pos += 1
        if p != "-":
            ppil4_pos += 1
        if c != "-" and p != "-":
            mapping[cypa_pos] = ppil4_pos

    return sorted(mapping[r] for r in active_site_cypa if r in mapping)


def write_actpass_file(active, passive, out_path):
    with open(out_path, "w") as f:
        f.write(" ".join(str(r) for r in active) + "\n")
        f.write(" ".join(str(r) for r in passive) + "\n")


def run(cmd, **kwargs):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def run_with_heartbeat(cmd, run_dir=None, step_plan=None, interval=20, **kwargs):
    """Like run(), but prints a live progress line every `interval` seconds
    while the subprocess is silent -- the complete routine's rigidbody/
    flexref/emref stages each churn through hundreds of models and can go
    quiet for a long time, which otherwise looks like it hung. If run_dir/
    step_plan are given, reports step name and % complete (see
    estimate_progress); otherwise just prints elapsed time."""
    print("+", " ".join(cmd))
    start = time.time()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(interval):
            elapsed = int(time.time() - start)
            if not (run_dir and step_plan):
                print(f"    ... still running ({elapsed}s elapsed)", flush=True)
                continue
            pct, current = estimate_progress(run_dir, step_plan)
            if current is None:
                print(f"    ... {elapsed}s elapsed, starting up (no step directory yet)", flush=True)
            else:
                idx, name, count, expected = current
                detail = f"{count}/{expected} models" if expected else "running"
                print(f"    ... {elapsed}s elapsed | step {idx + 1}/{len(step_plan)} "
                      f"({name}): {detail} | overall ~{pct * 100:.0f}%", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        subprocess.run(cmd, check=True, **kwargs)
    finally:
        stop.set()
        thread.join()


def print_table(rows, columns, title=None):
    if not rows:
        print("(no rows)")
        return
    get = lambda r, key: key(r) if callable(key) else str(r[key])
    widths = {label: max(len(label), *(len(get(r, key)) for r in rows)) for label, key in columns}
    if title:
        print(f"\n== {title} ==")
    header_line = "  ".join(label.ljust(widths[label]) for label, _ in columns)
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print("  ".join(get(r, key).ljust(widths[label]) for label, key in columns))


def read_final_capri_rows(haddock_run_dir):
    """Return (step_dir, rows) for the last caprieval step of a finished run, or (None, None)."""
    caprieval_dirs = sorted(
        glob.glob(os.path.join(haddock_run_dir, "[0-9]*_caprieval")),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    if not caprieval_dirs:
        return None, None

    final_dir = caprieval_dirs[-1]
    with open(os.path.join(final_dir, "capri_clt.tsv")) as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    header = lines[0].strip().split("\t")
    rows = [dict(zip(header, line.strip().split("\t"))) for line in lines[1:]]
    rows.sort(key=lambda r: int(r["caprieval_rank"]))
    return final_dir, rows


CAPRI_COLUMNS = [
    ("rank", "caprieval_rank"), ("cluster", "cluster_id"), ("n", "n"),
    ("score", "score"), ("dockq", "dockq"),
    ("irmsd", "irmsd"), ("fnat", "fnat"), ("lrmsd", "lrmsd"),
]


def print_capri_summary(haddock_run_dir):
    final_dir, rows = read_final_capri_rows(haddock_run_dir)
    if rows is None:
        print("No caprieval output found.")
        return
    print_table(rows, CAPRI_COLUMNS, title=f"Final CAPRI cluster results ({os.path.basename(final_dir)})")


def pick_candidate_name():
    if CANDIDATE_NAME:
        return CANDIDATE_NAME
    with open(SCREENING_SUMMARY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows in {SCREENING_SUMMARY_CSV} -- run 06 first, or set CANDIDATE_NAME directly.")
    best = min(rows, key=lambda r: float(r["combined_affinity"]))
    print(f"CANDIDATE_NAME not set -- auto-selecting '{best['name']}' (best combined_affinity "
          f"{float(best['combined_affinity']):.2f} kcal/mol) from 06's screening summary. "
          "Set CANDIDATE_NAME explicitly to run a different candidate, e.g. 07's best-dockq winner.")
    return best["name"]


def load_vina_affinities(candidate_name):
    if not os.path.exists(SCREENING_SUMMARY_CSV):
        return None
    with open(SCREENING_SUMMARY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["name"] == candidate_name:
                return row
    return None


def main():
    os.makedirs(RUN_DIR_BASE, exist_ok=True)
    candidate_name = pick_candidate_name()
    print(f"== Running COMPLETE HADDOCK3 routine for candidate: {candidate_name} ==")

    vina_row = load_vina_affinities(candidate_name)
    if vina_row:
        print(f"06's Vina screening: CRBN {float(vina_row['crbn_affinity']):.2f}, "
              f"PPIL4 {float(vina_row['ppil4_affinity']):.2f}, "
              f"combined {float(vina_row['combined_affinity']):.2f} kcal/mol")

    candidate_vina_dir = os.path.join(VINA_OUT_DIR, candidate_name)
    contacts_path = os.path.join(candidate_vina_dir, "crbn_contacts.txt")
    if not os.path.exists(contacts_path):
        sys.exit(f"{contacts_path} not found -- run 06 for this candidate first.")
    with open(contacts_path) as f:
        crbn_active = [int(x) for x in f.readline().split()]
    print("CRBN active (candidate-contact) residues:", crbn_active)

    print("== Renaming PPIL4 chain A -> B (HADDOCK3 requires unique chain/segids per partner) ==")
    with open(PPIL4_PDB, "w") as out:
        run(["pdb_chain", "-B", PPIL4_SOURCE_PDB], stdout=out)

    print("== Computing PPIL4 pocket residues (CypA-homology mapping) ==")
    ppil4_active = ppil4_pocket_residues()
    print("PPIL4 active (pocket) residues:", ppil4_active)

    print("== Deriving passive residues via haddock3-restraints ==")
    crbn_active_csv = ",".join(str(r) for r in crbn_active)
    # Use the receptor-only PDB (no ligand atoms) for passive_from_active --
    # the docked candidate isn't part of the CNS topology (see the
    # simplification noted in 05_haddock3_ternary_test_(Ryan).py).
    crbn_passive_out = subprocess.run(
        ["haddock3-restraints", "passive_from_active", CRBN_RECEPTOR_ONLY_PDB, crbn_active_csv, "-c", "A"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    crbn_passive = [int(x) for x in crbn_passive_out.split()] if crbn_passive_out else []
    print("CRBN passive residues:", crbn_passive)

    ppil4_active_csv = ",".join(str(r) for r in ppil4_active)
    ppil4_passive_out = subprocess.run(
        ["haddock3-restraints", "passive_from_active", PPIL4_PDB, ppil4_active_csv, "-c", "B"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    ppil4_passive = [int(x) for x in ppil4_passive_out.split()] if ppil4_passive_out else []
    print("PPIL4 passive residues:", ppil4_passive)

    candidate_run_dir = os.path.join(RUN_DIR_BASE, candidate_name)
    os.makedirs(candidate_run_dir, exist_ok=True)

    crbn_actpass = os.path.join(candidate_run_dir, "crbn_actpass.txt")
    write_actpass_file(crbn_active, crbn_passive, crbn_actpass)
    ppil4_actpass = os.path.join(candidate_run_dir, "ppil4_actpass.txt")
    write_actpass_file(ppil4_active, ppil4_passive, ppil4_actpass)

    print("== Generating ambig.tbl ==")
    ambig_tbl = os.path.join(candidate_run_dir, "ambig.tbl")
    with open(ambig_tbl, "w") as out:
        subprocess.run(
            ["haddock3-restraints", "active_passive_to_ambig", crbn_actpass, ppil4_actpass,
             "--segid-one", "A", "--segid-two", "B"],
            check=True, stdout=out,
        )
    print(f"Wrote {ambig_tbl}")

    print("== Writing HADDOCK3 config (complete routine: full sampling + emref + clustfcc) ==")
    haddock_run_dir = os.path.join(candidate_run_dir, "run1")
    cfg_path = os.path.join(candidate_run_dir, "haddock3_complete.toml")
    cfg = f"""
run_dir = "{haddock_run_dir}"
ncores = {NCORES}

molecules = [
    "{CRBN_RECEPTOR_ONLY_PDB}",
    "{PPIL4_PDB}"
]

[topoaa]

[rigidbody]
ambig_fname = "{ambig_tbl}"
sampling = 1000

[caprieval]

[seletop]
select = 200

[flexref]
ambig_fname = "{ambig_tbl}"

[caprieval]

[emref]
ambig_fname = "{ambig_tbl}"

[caprieval]

[clustfcc]

[caprieval]
"""
    with open(cfg_path, "w") as f:
        f.write(cfg.strip() + "\n")
    print(f"Wrote {cfg_path}")

    print("== Running complete HADDOCK3 routine (rough estimate: 45 min - 1.5 hr) ==")
    if os.path.exists(haddock_run_dir):
        print(f"Removing existing run_dir from a previous run: {haddock_run_dir}")
        shutil.rmtree(haddock_run_dir)
    run_with_heartbeat(["haddock3", cfg_path], run_dir=haddock_run_dir, step_plan=STEP_PLAN)

    print_capri_summary(haddock_run_dir)


if __name__ == "__main__":
    main()
