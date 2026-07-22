"""
HADDOCK3 ternary-complex test case: CRBN(+thalidomide-contact restraints) vs PPIL4.

This is a first structural test of the "CRBN-drug unit vs PPIL4" docking
approach discussed as an upgrade over the naive two-independent-RF composite
score. It uses Thalidomide as the test glue candidate because a real
CRBN-thalidomide crystal structure is available locally (giving CRBN's
thalidomide-induced conformation and true contact residues), unlike a novel
candidate where we'd have to blind-dock the glue into CRBN first.

Simplification for this first pass: thalidomide itself is NOT included as an
explicit ligand in the HADDOCK3 topology (CNS has no built-in topology for
it, and generating one via PRODRG/a custom .top+.param is a separate task).
Instead we keep CRBN in its thalidomide-bound conformation and restrain the
docking to the thalidomide-contacting face via AIR (ambiguous interaction
restraints) -- the CRBN surface that would present the drug's exposed
"molecular glue" epitope to PPIL4. This is the same simplification named
when we discussed HADDOCK3 vs. full ligand-aware docking.

PPIL4-side active residues reuse the exact CypA-homology pocket mapping
already used for Vina docking in 03_mgd_ppil4_crbn_pipeline_(Ryan).py, so
both docking approaches are restrained to the same PPIL4 site.
"""
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

REFERENCE_PDB = os.path.join(SCRIPT_DIR, "CRBN-Thalidomide-SALL4_(Ryan).pdb")
# NOTE: CNS's "@@" file-include syntax treats parentheses as special
# characters and truncates the path at "(" -- so any PDB/PSF path that CNS
# opens directly (i.e. anything passed to HADDOCK3 as a `molecules` entry)
# must NOT contain parentheses. Keep the "(Ryan)" suffix only for the
# committed reference file, not for these working/intermediate structures.
CRBN_RECEPTOR_PDB = os.path.join(SCRIPT_DIR, "CRBN_receptor_thalidomide_Ryan.pdb")
PPIL4_SOURCE_PDB = os.path.join(SCRIPT_DIR, "PPIL4_alphafold_(Ryan).pdb")
RUN_DIR = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_thalidomide_test")
PPIL4_PDB = os.path.join(RUN_DIR, "PPIL4_chainB.pdb")

CONTACT_CUTOFF = 4.5  # Angstrom, CRBN residue counted "active" if within this of thalidomide


def extract_crbn_chain_a(reference_pdb, out_pdb):
    """Keep only chain A protein atoms + its structural Zn (drop thalidomide, waters, chain B).

    The reference PDB (a PyMOL export) has every atom duplicated -- identical
    coordinates under a different atom serial number, all within one MODEL
    block -- so we dedupe on (chain, resnum, atom name), keeping the first
    occurrence.
    """
    kept = []
    seen = set()
    with open(reference_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21]
            resname = line[17:20].strip()
            if chain != "A":
                continue
            if line.startswith("HETATM") and resname != "ZN":
                continue  # drop EF2 (thalidomide) and HOH, keep the structural Zn
            atom_name = line[12:16].strip()
            resnum = line[22:26].strip()
            key = (chain, resnum, atom_name)
            if key in seen:
                continue
            seen.add(key)
            kept.append(line)
    with open(out_pdb, "w") as f:
        f.writelines(kept)
        f.write("END\n")
    return out_pdb


def find_thalidomide_contacts(reference_pdb, cutoff=CONTACT_CUTOFF):
    """Residues in chain A (CRBN) within `cutoff` Angstrom of any EF2 (thalidomide) atom."""
    ef2_coords = []
    protein_atoms = []  # (resnum, x, y, z)
    with open(reference_pdb) as f:
        for line in f:
            if line.startswith("HETATM") and line[17:20].strip() == "EF2" and line[21] == "A":
                ef2_coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            elif line.startswith("ATOM") and line[21] == "A":
                resnum = int(line[22:26])
                protein_atoms.append((resnum, float(line[30:38]), float(line[38:46]), float(line[46:54])))

    contacts = set()
    for resnum, x, y, z in protein_atoms:
        for ex, ey, ez in ef2_coords:
            d2 = (x - ex) ** 2 + (y - ey) ** 2 + (z - ez) ** 2
            if d2 <= cutoff ** 2:
                contacts.add(resnum)
                break
    return sorted(contacts)


def ppil4_pocket_residues():
    """Reproduce the exact CypA-homology active-site mapping used for the Vina PPIL4 receptor."""
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


def run_with_heartbeat(cmd, interval=20, **kwargs):
    """Like run(), but prints a heartbeat line every `interval` seconds while
    the subprocess is silent -- haddock3 goes quiet for tens of seconds to a
    few minutes during CNS computation, which otherwise looks like it hung."""
    print("+", " ".join(cmd))
    start = time.time()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(interval):
            print(f"    ... still running ({int(time.time() - start)}s elapsed)")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        subprocess.run(cmd, check=True, **kwargs)
    finally:
        stop.set()
        thread.join()


def print_capri_summary(haddock_run_dir):
    """Print the cluster stats from the last caprieval step of a finished run."""
    caprieval_dirs = sorted(
        glob.glob(os.path.join(haddock_run_dir, "[0-9]*_caprieval")),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    if not caprieval_dirs:
        print("No caprieval output found.")
        return

    final_dir = caprieval_dirs[-1]
    with open(os.path.join(final_dir, "capri_clt.tsv")) as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    header = lines[0].strip().split("\t")
    rows = [dict(zip(header, line.strip().split("\t"))) for line in lines[1:]]
    rows.sort(key=lambda r: int(r["caprieval_rank"]))

    columns = [
        ("rank", "caprieval_rank"), ("cluster", "cluster_id"), ("n", "n"),
        ("score", "score"), ("dockq", "dockq"),
        ("irmsd", "irmsd"), ("fnat", "fnat"), ("lrmsd", "lrmsd"),
    ]
    widths = {label: max(len(label), *(len(r[key]) for r in rows)) for label, key in columns}

    print(f"\n== Final CAPRI cluster results ({os.path.basename(final_dir)}) ==")
    header_line = "  ".join(label.ljust(widths[label]) for label, _ in columns)
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print("  ".join(r[key].ljust(widths[label]) for label, key in columns))


def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    print("== Extracting CRBN chain A (thalidomide-bound conformation) ==")
    extract_crbn_chain_a(REFERENCE_PDB, CRBN_RECEPTOR_PDB)

    print("== Renaming PPIL4 chain A -> B (HADDOCK3 requires unique chain/segids per partner) ==")
    with open(PPIL4_PDB, "w") as out:
        run(["pdb_chain", "-B", PPIL4_SOURCE_PDB], stdout=out)

    print("== Finding thalidomide-contact residues on CRBN ==")
    crbn_active = find_thalidomide_contacts(REFERENCE_PDB)
    print("CRBN active (thalidomide-contact) residues:", crbn_active)

    print("== Computing PPIL4 pocket residues (CypA-homology mapping) ==")
    ppil4_active = ppil4_pocket_residues()
    print("PPIL4 active (pocket) residues:", ppil4_active)

    print("== Deriving passive residues via haddock3-restraints ==")
    crbn_active_csv = ",".join(str(r) for r in crbn_active)
    ppil4_active_csv = ",".join(str(r) for r in ppil4_active)

    crbn_passive_out = subprocess.run(
        ["haddock3-restraints", "passive_from_active", CRBN_RECEPTOR_PDB, crbn_active_csv, "-c", "A"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    crbn_passive = [int(x) for x in crbn_passive_out.split()] if crbn_passive_out else []
    print("CRBN passive residues:", crbn_passive)

    ppil4_passive_out = subprocess.run(
        ["haddock3-restraints", "passive_from_active", PPIL4_PDB, ppil4_active_csv, "-c", "B"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    ppil4_passive = [int(x) for x in ppil4_passive_out.split()] if ppil4_passive_out else []
    print("PPIL4 passive residues:", ppil4_passive)

    crbn_actpass = os.path.join(RUN_DIR, "crbn_actpass.txt")
    ppil4_actpass = os.path.join(RUN_DIR, "ppil4_actpass.txt")
    write_actpass_file(crbn_active, crbn_passive, crbn_actpass)
    write_actpass_file(ppil4_active, ppil4_passive, ppil4_actpass)

    print("== Generating ambig.tbl ==")
    ambig_tbl = os.path.join(RUN_DIR, "ambig.tbl")
    with open(ambig_tbl, "w") as out:
        subprocess.run(
            ["haddock3-restraints", "active_passive_to_ambig", crbn_actpass, ppil4_actpass,
             "--segid-one", "A", "--segid-two", "B"],
            check=True, stdout=out,
        )
    print(f"Wrote {ambig_tbl}")

    print("== Writing HADDOCK3 config ==")
    haddock_run_dir = os.path.join(RUN_DIR, "run1")
    cfg_path = os.path.join(RUN_DIR, "haddock3_thalidomide_test.toml")
    cfg = f"""
run_dir = "{haddock_run_dir}"

molecules = [
    "{CRBN_RECEPTOR_PDB}",
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
    run_with_heartbeat(["haddock3", cfg_path])

    print_capri_summary(haddock_run_dir)
    print("== Done. See:", haddock_run_dir)


if __name__ == "__main__":
    main()
