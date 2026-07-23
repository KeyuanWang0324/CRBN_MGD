"""
EXPERIMENTAL: HADDOCK3 3-body ternary docking that includes the actual drug
molecule -- unlike 05/07/08 (which only ever dock CRBN against PPIL4, using
the candidate's Vina-derived contact residues as a proxy and discarding the
ligand's own atoms), this script docks THREE separate bodies together:
CRBN (chain A), the candidate ligand itself (chain C), and PPIL4 (chain B).
The ligand bridges the other two via real AIR restraints on both sides, so
its actual geometry/chemistry can influence whether/how the ternary complex
comes together -- not just a residue-list hint.

This closes the "drug-CRBN interaction" and "drug-PPIL4 interaction" gaps
discussed for 07/08 (see conversation). It does NOT address the third gap
(ligand-induced conformational change) beyond whatever local flexibility
flexref/emref already allow near the interface -- see this project's
existing caveins on that.

WHY THIS NEEDS ITS OWN SCRIPT (not just a flag on 08):
HADDOCK3's CNS engine needs a topology + force-field parameter file for any
non-standard residue (the ligand), which Vina/meeko's PDBQT route doesn't
produce. HADDOCK3 bundles PRODRG (haddock/prodrg/prodrg_<arch>) and can run
it automatically via `[topoaa] autotoppar = true` -- but testing that path
surfaced a real gap in HADDOCK3's own integration: PRODRG renames the
ligand's atoms to match its generated topology (e.g. "H1" -> "HG3"), and the
auto path never propagates that renaming back into the working PDB's
coordinates, so CNS aborts with "unknown coordinates for atom ...". (There
is literally a `# TODO: Check if the atom names have been changed!` comment
in haddock3's own libligand.py for this.) This script works around it by
running PRODRG once per candidate itself, keeping PRODRG's own renamed
coordinate file (DRGFIN.PDB) instead of the original input coordinates, and
passing the resulting topology/param explicitly via `ligand_top_fname`/
`ligand_param_fname` -- which must be set on EVERY module that runs its own
CNS minimization (topoaa, rigidbody, flexref, emref), not just topoaa, or
CNS aborts with "missing nonbonded Lennard-Jones parameters" instead.

This imports a few non-public haddock.libs functions (PRODRG binary
resolution, atom-name/NBONDS sanitization) rather than reimplementing them,
so behavior stays exactly consistent with whatever HADDOCK3 version is
installed -- but that does mean this script is more sensitive to a
haddock3 upgrade than 05/07/08 (which only ever shell out to the stable
`haddock3`/`haddock3-restraints` CLIs).

Validated with a 4-model smoke test (rigidbody -> caprieval -> seletop ->
flexref -> caprieval -> emref -> caprieval -> clustfcc -> caprieval) on
cand_5 before writing this: completed in 74s, top cluster n=3/4, dockq
0.627. This script runs the same chain at 08's full scale (sampling=1000,
top 200 for flexref/emref).

Run in the haddock3 venv:
    source .venv-haddock3/bin/activate
    python3 "10_haddock3_ternary_with_ligand_(Ryan).py"
"""
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

SCRIPT_START_TIME = time.time()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if shutil.which("haddock3") is None:
    haddock_venv_bin = os.path.join(SCRIPT_DIR, ".venv-haddock3", "bin")
    haddock_python = os.path.join(haddock_venv_bin, "python3")
    env = os.environ.copy()
    env["PATH"] = haddock_venv_bin + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = os.path.join(SCRIPT_DIR, ".venv-haddock3")
    os.execve(haddock_python, [haddock_python] + sys.argv, env)

from haddock.libs.libutil import get_prodrg_exec

VINA_OUT_DIR = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_novel_candidate")
TERNARY_SCORES_CSV = os.path.join(SCRIPT_DIR, "08_final_ternary_results_(Ryan).csv")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "10_final_ternary_with_ligand_results_(Ryan).csv")

# Set this to a specific candidate's name to run only that one. Leave blank
# to auto-pick the best-dockq candidate from 08's results.
CANDIDATE_NAME = ""

CRBN_RECEPTOR_ONLY_PDB = os.path.join(SCRIPT_DIR, "CRBN_receptor_thalidomide_Ryan.pdb")
PPIL4_SOURCE_PDB = os.path.join(SCRIPT_DIR, "PPIL4_alphafold_(Ryan).pdb")
RUN_DIR_BASE = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_ternary_with_ligand_run")
PPIL4_PDB = os.path.join(RUN_DIR_BASE, "PPIL4_chainB.pdb")

NCORES = max(1, (os.cpu_count() or 4) - 1)

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


def write_actpass_file(active, passive, out_path):
    with open(out_path, "w") as f:
        f.write(" ".join(str(r) for r in active) + "\n")
        f.write(" ".join(str(r) for r in passive) + "\n")


def run(cmd, **kwargs):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def run_with_heartbeat(cmd, run_dir=None, step_plan=None, interval=20, label=None, **kwargs):
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
    if not os.path.exists(TERNARY_SCORES_CSV):
        sys.exit(f"{TERNARY_SCORES_CSV} not found -- run 08 first, or set CANDIDATE_NAME directly.")
    with open(TERNARY_SCORES_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dockq"] != "-"]
    if not rows:
        sys.exit(f"No usable dockq rows in {TERNARY_SCORES_CSV} -- set CANDIDATE_NAME directly.")
    best = max(rows, key=lambda r: float(r["dockq"]))
    print(f"CANDIDATE_NAME not set -- auto-selecting '{best['name']}' (best dockq "
          f"{float(best['dockq']):.3f}) from {TERNARY_SCORES_CSV}.")
    return best["name"]


def extract_ligand_pdb(candidate_name, out_path):
    """Pull just the ligand (resn LIG, chain C) HETATM lines out of 06's
    CRBN+ligand complex into a standalone PDB."""
    complex_path = os.path.join(VINA_OUT_DIR, candidate_name, "CRBN_candidate_complex.pdb")
    if not os.path.exists(complex_path):
        sys.exit(f"{complex_path} not found -- run 06 for this candidate first.")
    with open(complex_path) as f:
        lig_lines = [l for l in f if l.startswith("HETATM") and l[17:20] == "LIG"]
    if not lig_lines:
        sys.exit(f"No LIG HETATM records found in {complex_path}.")
    with open(out_path, "w") as f:
        f.writelines(lig_lines)
        f.write("END\n")


def _sanitize_atom_names(content):
    """Mirrors haddock.libs.libligand._sanitize_atom_names: PRODRG can emit
    atom type names containing colons (e.g. "HT:A"), which CNS rejects."""
    lines = []
    for line in content.splitlines(keepends=True):
        if line.lstrip().startswith("!"):
            lines.append(line)
        else:
            lines.append(line.replace(":", ""))
    return "".join(lines)


def _remove_nbonds(par_content):
    """Mirrors haddock.libs.libligand._remove_nbonds: PRODRG's own NBONds
    block can conflict with HADDOCK's internal nonbonded parameters."""
    return re.sub(r"(?s)NBONds.*?END", "", par_content)


def run_prodrg_for_ligand(ligand_pdb, out_dir):
    """Run PRODRG on a ligand PDB and return (top_path, param_path,
    fixed_pdb_path) -- CNS topology, parameters, and a coordinate PDB whose
    atom names actually match that topology (chain C, resnum 900, matching
    this project's convention), all written into out_dir.

    Unlike haddock.libs.libligand.run_prodrg (which discards PRODRG's own
    renamed-atom PDB, DRGFIN.PDB), this keeps it -- see this file's
    docstring for why that renaming matters."""
    prodrg_exec, prodrg_param = get_prodrg_exec()
    if prodrg_exec is None:
        sys.exit("PRODRG binary not found for this platform (haddock.libs.libutil.get_prodrg_exec "
                  "returned None) -- this script requires the bundled PRODRG binary to parametrize "
                  "the ligand.")

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(prodrg_param, os.path.join(tmpdir, os.path.basename(prodrg_param)))
        shutil.copy(ligand_pdb, os.path.join(tmpdir, os.path.basename(ligand_pdb)))

        result = subprocess.run(
            [str(prodrg_exec), os.path.basename(ligand_pdb), os.path.basename(prodrg_param), "PDBELEM"],
            cwd=tmpdir, capture_output=True, text=True,
        )
        if result.returncode != 0:
            sys.exit(f"PRODRG failed:\n{result.stderr}")

        tmp_top = os.path.join(tmpdir, "DRGCNS.TOP")
        tmp_par = os.path.join(tmpdir, "DRGCNS.PAR")
        tmp_pdb = os.path.join(tmpdir, "DRGFIN.PDB")
        if not (os.path.exists(tmp_top) and os.path.exists(tmp_par) and os.path.exists(tmp_pdb)):
            sys.exit(f"PRODRG finished but expected output files are missing in {tmpdir}: "
                      f"{sorted(os.listdir(tmpdir))}")

        stem = os.path.splitext(os.path.basename(ligand_pdb))[0]
        top_path = os.path.join(out_dir, f"{stem}_prodrg.top")
        par_path = os.path.join(out_dir, f"{stem}_prodrg.param")
        fixed_pdb_path = os.path.join(out_dir, f"{stem}_prodrg_fixed.pdb")

        with open(tmp_top) as f:
            top_content = _sanitize_atom_names(f.read())
        with open(tmp_par) as f:
            par_content = _sanitize_atom_names(_remove_nbonds(f.read()))
        with open(tmp_pdb) as f:
            pdb_lines = f.readlines()

        with open(top_path, "w") as f:
            f.write(top_content)
        with open(par_path, "w") as f:
            f.write(par_content)
        with open(fixed_pdb_path, "w") as f:
            for line in pdb_lines:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    line = line[:21] + "C" + " 900" + line[26:]
                f.write(line)
            if not pdb_lines[-1].startswith("END"):
                f.write("END\n")

    return top_path, par_path, fixed_pdb_path


def main():
    os.makedirs(RUN_DIR_BASE, exist_ok=True)
    candidate_name = pick_candidate_name()
    candidate_run_dir = os.path.join(RUN_DIR_BASE, candidate_name)
    os.makedirs(candidate_run_dir, exist_ok=True)
    print(f"[{candidate_name}] step 0/{len(STEP_PLAN)}: preparing ligand topology/restraints")

    # --- PPIL4-side setup (candidate-independent) ---
    with open(PPIL4_PDB, "w") as out:
        run(["pdb_chain", "-B", PPIL4_SOURCE_PDB], stdout=out)
    ppil4_active = [44, 49, 50, 52, 60, 97, 98, 99, 107, 109, 118, 119, 123]  # CypA-homology pocket, see 05/07/08
    ppil4_active_csv = ",".join(str(r) for r in ppil4_active)
    ppil4_passive_out = subprocess.run(
        ["haddock3-restraints", "passive_from_active", PPIL4_PDB, ppil4_active_csv, "-c", "B"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    ppil4_passive = [int(x) for x in ppil4_passive_out.split()] if ppil4_passive_out else []
    ppil4_actpass = os.path.join(candidate_run_dir, "ppil4_actpass.txt")
    write_actpass_file(ppil4_active, ppil4_passive, ppil4_actpass)

    # --- CRBN-side setup (from 06's Vina contact residues) ---
    contacts_path = os.path.join(VINA_OUT_DIR, candidate_name, "crbn_contacts.txt")
    if not os.path.exists(contacts_path):
        sys.exit(f"{contacts_path} not found -- run 06 for this candidate first.")
    with open(contacts_path) as f:
        crbn_active = [int(x) for x in f.readline().split()]
    crbn_active_csv = ",".join(str(r) for r in crbn_active)
    crbn_passive_out = subprocess.run(
        ["haddock3-restraints", "passive_from_active", CRBN_RECEPTOR_ONLY_PDB, crbn_active_csv, "-c", "A"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    crbn_passive = [int(x) for x in crbn_passive_out.split()] if crbn_passive_out else []
    crbn_actpass = os.path.join(candidate_run_dir, "crbn_actpass.txt")
    write_actpass_file(crbn_active, crbn_passive, crbn_actpass)

    # --- Ligand: extract, parametrize via PRODRG, and mark the whole thing "active" ---
    ligand_raw_pdb = os.path.join(candidate_run_dir, "ligand.pdb")
    extract_ligand_pdb(candidate_name, ligand_raw_pdb)
    ligand_top, ligand_param, ligand_pdb = run_prodrg_for_ligand(ligand_raw_pdb, candidate_run_dir)
    ligand_actpass = os.path.join(candidate_run_dir, "ligand_actpass.txt")
    write_actpass_file([900], [], ligand_actpass)  # the whole ligand is one CNS residue (resnum 900)

    # --- Restraints: ligand<->CRBN and ligand<->PPIL4, concatenated into one ambig.tbl ---
    crbn_ligand_tbl = subprocess.run(
        ["haddock3-restraints", "active_passive_to_ambig", crbn_actpass, ligand_actpass,
         "--segid-one", "A", "--segid-two", "C"],
        check=True, capture_output=True, text=True,
    ).stdout
    ppil4_ligand_tbl = subprocess.run(
        ["haddock3-restraints", "active_passive_to_ambig", ppil4_actpass, ligand_actpass,
         "--segid-one", "B", "--segid-two", "C"],
        check=True, capture_output=True, text=True,
    ).stdout
    ambig_tbl = os.path.join(candidate_run_dir, "ambig.tbl")
    with open(ambig_tbl, "w") as f:
        f.write(crbn_ligand_tbl)
        f.write(ppil4_ligand_tbl)

    # --- HADDOCK3 config: 3 molecules, ligand_top_fname/ligand_param_fname repeated on
    # every module that runs its own CNS minimization (topoaa, rigidbody, flexref, emref) ---
    haddock_run_dir = os.path.join(candidate_run_dir, "run1")
    cfg_path = os.path.join(candidate_run_dir, "haddock3_ternary_with_ligand.toml")
    cfg = f"""
run_dir = "{haddock_run_dir}"
ncores = {NCORES}

molecules = [
    "{CRBN_RECEPTOR_ONLY_PDB}",
    "{ligand_pdb}",
    "{PPIL4_PDB}"
]

[topoaa]
ligand_top_fname = "{ligand_top}"
ligand_param_fname = "{ligand_param}"

[rigidbody]
ambig_fname = "{ambig_tbl}"
ligand_top_fname = "{ligand_top}"
ligand_param_fname = "{ligand_param}"
sampling = 1000

[caprieval]

[seletop]
select = 200

[flexref]
ambig_fname = "{ambig_tbl}"
ligand_top_fname = "{ligand_top}"
ligand_param_fname = "{ligand_param}"

[caprieval]

[emref]
ambig_fname = "{ambig_tbl}"
ligand_top_fname = "{ligand_top}"
ligand_param_fname = "{ligand_param}"

[caprieval]

[clustfcc]

[caprieval]
"""
    with open(cfg_path, "w") as f:
        f.write(cfg.strip() + "\n")

    if os.path.exists(haddock_run_dir):
        shutil.rmtree(haddock_run_dir)
    print(f"[{candidate_name}] starting HADDOCK3 3-body (CRBN+ligand+PPIL4) run "
          f"(rough estimate: 45 min - 1.5 hr, similar to 08)")
    run_with_heartbeat(["haddock3", cfg_path], run_dir=haddock_run_dir, step_plan=STEP_PLAN, label=candidate_name)

    print_capri_summary(haddock_run_dir)
    _, rows = read_final_capri_rows(haddock_run_dir)
    if rows:
        top_row = rows[0]
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "cluster_rank", "cluster_id", "n", "score", "dockq", "irmsd", "fnat", "lrmsd"])
            writer.writerow([candidate_name, top_row["caprieval_rank"], top_row["cluster_id"], top_row["n"],
                              top_row["score"], top_row["dockq"], top_row["irmsd"], top_row["fnat"],
                              top_row["lrmsd"]])
        print(f"Wrote {RESULTS_CSV}")

    total = time.time() - SCRIPT_START_TIME
    print(f"Total script runtime: {total:.0f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
