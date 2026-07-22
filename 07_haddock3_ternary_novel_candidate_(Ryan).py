"""
HADDOCK3 ternary-complex docking for novel CRBN-glue candidates vs PPIL4.

Follow-on to 05_haddock3_ternary_test_(Ryan).py (the Thalidomide reference-
structure test case) and 06_dock_candidate_crbn_(Ryan).py (which Vina-screens
candidates into CRBN's pocket and ranks them by affinity, since -- unlike
Thalidomide -- no crystal structure exists for them). This script reads that
screening ranking, derives CRBN-side AIR restraints from each top candidate's
docked-ligand contact residues (in place of thalidomide's crystallographic
contacts), and runs the full (slow) HADDOCK3 ternary docking against PPIL4,
using the same CypA-homology pocket restraints as before, for only the
top TOP_N candidates.

Run in the haddock3 venv:
    source .venv-haddock3/bin/activate
    python3 "07_haddock3_ternary_novel_candidate_(Ryan).py"
"""
import glob
import os
import shutil
import subprocess
import sys

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
SCREENING_SUMMARY_TSV = os.path.join(VINA_OUT_DIR, "screening_summary.tsv")

# Only the top TOP_N Vina-screened candidates get the full (~3+ min each)
# HADDOCK3 ternary treatment; raise/lower as needed.
TOP_N = 5

# CNS's "@@" include syntax truncates paths at "(" -- keep this filename
# parenthesis-free since it's fed directly to HADDOCK3 as a molecule.
CRBN_RECEPTOR_ONLY_PDB = os.path.join(SCRIPT_DIR, "CRBN_receptor_thalidomide_Ryan.pdb")
PPIL4_SOURCE_PDB = os.path.join(SCRIPT_DIR, "PPIL4_alphafold_(Ryan).pdb")
RUN_DIR_BASE = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_novel_candidate_run")
PPIL4_PDB = os.path.join(RUN_DIR_BASE, "PPIL4_chainB.pdb")


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


def dock_one_candidate(candidate_name, crbn_affinity, ppil4_affinity, combined_affinity, crbn_active, ppil4_actpass):
    candidate_run_dir = os.path.join(RUN_DIR_BASE, candidate_name)
    os.makedirs(candidate_run_dir, exist_ok=True)

    print("== Deriving CRBN passive residues via haddock3-restraints ==")
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

    crbn_actpass = os.path.join(candidate_run_dir, "crbn_actpass.txt")
    write_actpass_file(crbn_active, crbn_passive, crbn_actpass)

    print("== Generating ambig.tbl ==")
    ambig_tbl = os.path.join(candidate_run_dir, "ambig.tbl")
    with open(ambig_tbl, "w") as out:
        subprocess.run(
            ["haddock3-restraints", "active_passive_to_ambig", crbn_actpass, ppil4_actpass,
             "--segid-one", "A", "--segid-two", "B"],
            check=True, stdout=out,
        )
    print(f"Wrote {ambig_tbl}")

    print("== Writing HADDOCK3 config ==")
    haddock_run_dir = os.path.join(candidate_run_dir, "run1")
    cfg_path = os.path.join(candidate_run_dir, "haddock3_novel_candidate.toml")
    cfg = f"""
run_dir = "{haddock_run_dir}"

molecules = [
    "{CRBN_RECEPTOR_ONLY_PDB}",
    "{PPIL4_PDB}"
]

[topoaa]

[rigidbody]
ambig_fname = "{ambig_tbl}"
sampling = 20

[caprieval]

[seletop]
select = 10

[flexref]
ambig_fname = "{ambig_tbl}"

[caprieval]
"""
    with open(cfg_path, "w") as f:
        f.write(cfg.strip() + "\n")
    print(f"Wrote {cfg_path}")

    print("== Running HADDOCK3 (this will take a while) ==")
    if os.path.exists(haddock_run_dir):
        print(f"Removing existing run_dir from a previous run: {haddock_run_dir}")
        shutil.rmtree(haddock_run_dir)
    run(["haddock3", cfg_path])

    print_capri_summary(haddock_run_dir)

    _, rows = read_final_capri_rows(haddock_run_dir)
    top_row = rows[0] if rows else None
    return {
        "name": candidate_name,
        "crbn_affinity": crbn_affinity,
        "ppil4_affinity": ppil4_affinity,
        "combined_affinity": combined_affinity,
        "score": top_row["score"] if top_row else "-",
        "dockq": top_row["dockq"] if top_row else "-",
        "irmsd": top_row["irmsd"] if top_row else "-",
        "fnat": top_row["fnat"] if top_row else "-",
        "lrmsd": top_row["lrmsd"] if top_row else "-",
    }


def main():
    os.makedirs(RUN_DIR_BASE, exist_ok=True)

    print("== Reading Vina screening results from 06 ==")
    with open(SCREENING_SUMMARY_TSV) as f:
        f.readline()  # header
        screened = [line.strip().split("\t") for line in f if line.strip()]
    screened = [
        {"name": name, "crbn_affinity": float(crbn), "ppil4_affinity": float(ppil4), "combined_affinity": float(combined)}
        for name, crbn, ppil4, combined in screened
    ]
    screened.sort(key=lambda r: r["combined_affinity"])

    selected = screened[:TOP_N]
    skipped = screened[TOP_N:]
    print(f"Running full HADDOCK3 ternary docking on top {len(selected)} of {len(screened)} screened candidates: "
          f"{', '.join(r['name'] for r in selected)}")
    if skipped:
        print(f"Skipping {len(skipped)} lower-ranked candidates: {', '.join(r['name'] for r in skipped)}")

    print("== Renaming PPIL4 chain A -> B (HADDOCK3 requires unique chain/segids per partner) ==")
    with open(PPIL4_PDB, "w") as out:
        run(["pdb_chain", "-B", PPIL4_SOURCE_PDB], stdout=out)

    print("== Computing PPIL4 pocket residues (CypA-homology mapping) ==")
    ppil4_active = ppil4_pocket_residues()
    print("PPIL4 active (pocket) residues:", ppil4_active)

    print("== Deriving PPIL4 passive residues via haddock3-restraints ==")
    ppil4_active_csv = ",".join(str(r) for r in ppil4_active)
    ppil4_passive_out = subprocess.run(
        ["haddock3-restraints", "passive_from_active", PPIL4_PDB, ppil4_active_csv, "-c", "B"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    ppil4_passive = [int(x) for x in ppil4_passive_out.split()] if ppil4_passive_out else []
    print("PPIL4 passive residues:", ppil4_passive)

    ppil4_actpass = os.path.join(RUN_DIR_BASE, "ppil4_actpass.txt")
    write_actpass_file(ppil4_active, ppil4_passive, ppil4_actpass)

    results = []
    for candidate in selected:
        print(f"\n== Candidate: {candidate['name']} "
              f"(Vina: CRBN {candidate['crbn_affinity']:.2f}, PPIL4 {candidate['ppil4_affinity']:.2f}, "
              f"combined {candidate['combined_affinity']:.2f} kcal/mol) ==")
        candidate_vina_dir = os.path.join(VINA_OUT_DIR, candidate["name"])
        with open(os.path.join(candidate_vina_dir, "crbn_contacts.txt")) as f:
            crbn_active = [int(x) for x in f.readline().split()]
        print("CRBN active (candidate-contact) residues:", crbn_active)

        result = dock_one_candidate(
            candidate["name"], candidate["crbn_affinity"], candidate["ppil4_affinity"],
            candidate["combined_affinity"], crbn_active, ppil4_actpass,
        )
        results.append(result)

    results.sort(key=lambda r: (r["dockq"] == "-", -float(r["dockq"]) if r["dockq"] != "-" else 0))
    print_table(
        results,
        [("name", "name"),
         ("crbn (kcal/mol)", lambda r: f"{r['crbn_affinity']:.2f}"),
         ("ppil4 (kcal/mol)", lambda r: f"{r['ppil4_affinity']:.2f}"),
         ("score", "score"), ("dockq", "dockq"), ("irmsd", "irmsd"),
         ("fnat", "fnat"), ("lrmsd", "lrmsd")],
        title="Candidate comparison (best dockq first)",
    )


if __name__ == "__main__":
    main()
