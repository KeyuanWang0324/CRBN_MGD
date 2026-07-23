"""
COMPLETE HADDOCK3 ternary-docking routine (full sampling + emref + clustfcc)
for the top-ranked CRBN-glue candidates vs PPIL4.

06_vina_dock_candidates_(Ryan).py and 07_haddock3_ternary_novel_candidate_(Ryan).py
run a cheap, truncated HADDOCK3 config (rigidbody sampling=20, flexref on the
top 10, no emref, no clustering) to rank many candidates quickly. This script
runs the COMPLETE stock HADDOCK3 protein-protein routine instead -- rigidbody
sampling=1000, flexref + emref refinement on the top 200, then clustfcc
clustering -- on the top TOP_N candidates (best dockq) from 07's comparison
table, one after another. Set CANDIDATE_NAME to run a single specific
candidate instead.

This is far more expensive than 06/07 (rough estimate: 45 min - 1.5 hr per
candidate on an 18-core machine, vs. ~3 min for 07's lite pass) -- keep
TOP_N small (a handful of finalists), not a full screen.

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

SCRIPT_START_TIME = time.time()
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
# Used only for the crbn_contacts.txt each candidate's restraints are built
# from -- not for auto-picking candidates, see pick_candidate_names() below.
VINA_SCREENING_CSV = os.path.join(SCRIPT_DIR, "06_vina_screening_scores_for_07_(Ryan).csv")
# 07's ranked output (best dockq first) -- the primary source for auto-
# picking a finalist when CANDIDATE_NAME is left blank.
TERNARY_SCORES_CSV = os.path.join(SCRIPT_DIR, "07_ternary_docking_scores_for_08_(Ryan).csv")
# This script's own output -- the finalist's final CAPRI cluster results.
# Terminal file (nothing downstream reads it), so no "_for_XX" suffix.
RESULTS_CSV = os.path.join(SCRIPT_DIR, "08_final_ternary_results_(Ryan).csv")

# Set this to a specific candidate's name from 07's comparison table to run
# only that one (bypasses TOP_N entirely). Leave blank to auto-pick the top
# TOP_N candidates by dockq from TERNARY_SCORES_CSV instead (falling back to
# 06's best combined_affinity if 07 hasn't been run yet).
CANDIDATE_NAME = ""

# How many of 07's top-ranked (by dockq) candidates to run the complete
# routine on, when CANDIDATE_NAME is left blank. Each candidate costs
# ~45 min - 1.5 hr, so raise/lower with total run time in mind.
TOP_N = 5

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


def run_with_heartbeat(cmd, run_dir=None, step_plan=None, interval=20, label=None, **kwargs):
    """Like run(), but prints a live progress line every `interval` seconds
    while the subprocess is silent -- the complete routine's rigidbody/
    flexref/emref stages each churn through hundreds of models and can go
    quiet for a long time, which otherwise looks like it hung. If run_dir/
    step_plan are given, reports step name and % complete (see
    estimate_progress); otherwise just prints elapsed time. `label` (e.g. a
    candidate name) is prefixed on each line.

    HADDOCK3's own console output (the startup banner, per-module INFO
    lines) is redirected to a log file next to run_dir instead of showing
    here, so only our concise status lines print. On failure, the log's
    path and tail are printed for diagnosis before re-raising."""
    prefix = f"[{label}] " if label else ""
    start = time.time()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(interval):
            elapsed = int(time.time() - start)
            if not (run_dir and step_plan):
                print(f"    ... {prefix}still running ({elapsed}s elapsed)", flush=True)
                continue
            pct, current = estimate_progress(run_dir, step_plan)
            if current is None:
                print(f"    ... {prefix}{elapsed}s elapsed, starting up", flush=True)
            else:
                idx, name, count, expected = current
                detail = f"{count}/{expected} models" if expected else "running"
                print(f"    ... {prefix}step {idx + 1}/{len(step_plan)} ({name}): "
                      f"{detail} | overall ~{pct * 100:.0f}%", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    log_path = os.path.join(os.path.dirname(run_dir), "haddock3_stdout.log") if run_dir else None
    try:
        if log_path:
            with open(log_path, "w") as log_f:
                subprocess.run(cmd, check=True, stdout=log_f, stderr=subprocess.STDOUT, **kwargs)
        else:
            subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError:
        if log_path:
            print(f"{prefix}HADDOCK3 failed -- last output from {log_path}:")
            with open(log_path) as log_f:
                print("".join(log_f.readlines()[-30:]))
        raise
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


def pick_candidate_names():
    if CANDIDATE_NAME:
        return [CANDIDATE_NAME]

    if os.path.exists(TERNARY_SCORES_CSV):
        with open(TERNARY_SCORES_CSV, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["dockq"] != "-"]
        if rows:
            rows.sort(key=lambda r: float(r["dockq"]), reverse=True)
            names = [r["name"] for r in rows[:TOP_N]]
            print(f"CANDIDATE_NAME not set -- auto-selecting top {len(names)} by dockq from "
                  f"{TERNARY_SCORES_CSV}: {', '.join(names)}")
            return names
        print(f"{TERNARY_SCORES_CSV} has no usable dockq rows -- falling back to 06's screening scores.")

    if not os.path.exists(VINA_SCREENING_CSV):
        sys.exit(f"Neither {TERNARY_SCORES_CSV} nor {VINA_SCREENING_CSV} found -- "
                  "run 06 (then 07) first, or set CANDIDATE_NAME directly.")
    with open(VINA_SCREENING_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows in {VINA_SCREENING_CSV} -- run 06 first, or set CANDIDATE_NAME directly.")
    rows.sort(key=lambda r: float(r["combined_affinity"]))
    names = [r["name"] for r in rows[:TOP_N]]
    print(f"CANDIDATE_NAME not set -- auto-selecting top {len(names)} by combined_affinity from "
          "06's screening scores -- run 07 for a dockq-based pick instead. "
          "Set CANDIDATE_NAME explicitly to run a single specific candidate.")
    return names


def write_results_csv(all_rows):
    """all_rows: list of (candidate_name, top_capri_row_or_None) pairs, one
    per candidate (its rank-1, best-score cluster -- not every cluster),
    accumulated as candidates finish. Written after every candidate so a
    later candidate's failure can't lose earlier candidates' results."""
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "cluster_rank", "cluster_id", "n", "score", "dockq", "irmsd", "fnat", "lrmsd"])
        for candidate_name, r in all_rows:
            if r is None:
                writer.writerow([candidate_name, "-", "-", "-", "-", "-", "-", "-", "-"])
            else:
                writer.writerow([candidate_name, r["caprieval_rank"], r["cluster_id"], r["n"],
                                  r["score"], r["dockq"], r["irmsd"], r["fnat"], r["lrmsd"]])
    print(f"Wrote {RESULTS_CSV}")


def run_candidate(candidate_name, ppil4_actpass, label):
    """Run the complete HADDOCK3 routine for one candidate against the
    (already-computed, candidate-independent) PPIL4 restraints. Returns the
    top (rank-1, best-score) cluster row, or None if setup/HADDOCK3 fails or
    no cluster was found."""
    print(f"[{label}] step 0/{len(STEP_PLAN)}: preparing restraints/config")

    candidate_vina_dir = os.path.join(VINA_OUT_DIR, candidate_name)
    contacts_path = os.path.join(candidate_vina_dir, "crbn_contacts.txt")
    if not os.path.exists(contacts_path):
        print(f"[{label}] {contacts_path} not found -- run 06 for this candidate first. Skipping.")
        return None
    with open(contacts_path) as f:
        crbn_active = [int(x) for x in f.readline().split()]

    crbn_active_csv = ",".join(str(r) for r in crbn_active)
    # Use the receptor-only PDB (no ligand atoms) for passive_from_active --
    # the docked candidate isn't part of the CNS topology (see the
    # simplification noted in 05_haddock3_ternary_test_(Ryan).py).
    crbn_passive_out = subprocess.run(
        ["haddock3-restraints", "passive_from_active", CRBN_RECEPTOR_ONLY_PDB, crbn_active_csv, "-c", "A"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    crbn_passive = [int(x) for x in crbn_passive_out.split()] if crbn_passive_out else []

    candidate_run_dir = os.path.join(RUN_DIR_BASE, candidate_name)
    os.makedirs(candidate_run_dir, exist_ok=True)

    crbn_actpass = os.path.join(candidate_run_dir, "crbn_actpass.txt")
    write_actpass_file(crbn_active, crbn_passive, crbn_actpass)

    ambig_tbl = os.path.join(candidate_run_dir, "ambig.tbl")
    with open(ambig_tbl, "w") as out:
        subprocess.run(
            ["haddock3-restraints", "active_passive_to_ambig", crbn_actpass, ppil4_actpass,
             "--segid-one", "A", "--segid-two", "B"],
            check=True, stdout=out,
        )

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

    if os.path.exists(haddock_run_dir):
        shutil.rmtree(haddock_run_dir)
    print(f"[{label}] starting HADDOCK3 (rough estimate: 45 min - 1.5 hr)")
    run_with_heartbeat(["haddock3", cfg_path], run_dir=haddock_run_dir, step_plan=STEP_PLAN, label=label)

    print_capri_summary(haddock_run_dir)
    _, rows = read_final_capri_rows(haddock_run_dir)
    return rows[0] if rows else None


def main():
    os.makedirs(RUN_DIR_BASE, exist_ok=True)
    candidate_names = pick_candidate_names()
    print(f"Running the complete HADDOCK3 routine on {len(candidate_names)} candidate(s): "
          f"{', '.join(candidate_names)}")

    # PPIL4-side restraints don't depend on the candidate -- compute once,
    # not once per candidate.
    with open(PPIL4_PDB, "w") as out:
        run(["pdb_chain", "-B", PPIL4_SOURCE_PDB], stdout=out)
    ppil4_active = ppil4_pocket_residues()
    ppil4_active_csv = ",".join(str(r) for r in ppil4_active)
    ppil4_passive_out = subprocess.run(
        ["haddock3-restraints", "passive_from_active", PPIL4_PDB, ppil4_active_csv, "-c", "B"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    ppil4_passive = [int(x) for x in ppil4_passive_out.split()] if ppil4_passive_out else []
    ppil4_actpass = os.path.join(RUN_DIR_BASE, "ppil4_actpass.txt")
    write_actpass_file(ppil4_active, ppil4_passive, ppil4_actpass)

    all_rows = []
    candidates_loop_start = time.time()
    for i, candidate_name in enumerate(candidate_names, 1):
        remaining = len(candidate_names) - i
        elapsed_so_far = time.time() - candidates_loop_start
        avg_per_candidate = elapsed_so_far / (i - 1) if i > 1 else None
        eta_str = (f"~{avg_per_candidate * remaining / 60:.0f} min remaining for the run"
                   if avg_per_candidate else "remaining time unknown until candidate 1 finishes")
        label = f"{candidate_name}, {i}/{len(candidate_names)}, {remaining} left"
        print(f"\n[{label}] ({elapsed_so_far / 60:.1f} min elapsed this run, {eta_str} | "
              f"{(time.time() - SCRIPT_START_TIME) / 60:.1f} min total script time)")

        try:
            top_row = run_candidate(candidate_name, ppil4_actpass, label)
        except subprocess.CalledProcessError:
            print(f"[{label}] HADDOCK3 failed for this candidate -- skipping it and continuing with the rest.")
            top_row = None

        all_rows.append((candidate_name, top_row))
        write_results_csv(all_rows)

    total = time.time() - SCRIPT_START_TIME
    print(f"\nTotal script runtime: {total:.0f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
