"""
Dock multiple candidate CRBN-glue candidates into both CRBN's thalidomide
pocket and PPIL4's CypA-homology pocket via Vina, as a fast pre-filter for
the HADDOCK3 ternary-docking step (see 07_haddock3_ternary_novel_candidate_(Ryan).py).

Run with the SYSTEM python (has vina/meeko/rdkit installed), not the
haddock3 venv:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
        "06_dock_candidate_crbn_(Ryan).py"

CRBN receptor: CRBN_receptor_thalidomide_Ryan.pdb (CRBN chain A, apo of
ligand, built in 05_haddock3_ternary_test_(Ryan).py). Box is centered on
where thalidomide sits in the reference crystal structure
(CRBN-Thalidomide-SALL4_(Ryan).pdb), same pocket, since each candidate
shares the identical glutarimide-isoindolinone CRBN-binding degron.

PPIL4 receptor: PPIL4_alphafold_(Ryan).pdb, box centered on the same
CypA-homology active-site residues used for restraint generation in 05/07.
NOTE: this is a simplification, not a mechanistic model -- in a real
molecular-glue ternary complex the small molecule typically stays bound to
CRBN and presents a new protein-protein interface to the neo-substrate,
rather than independently occupying a pocket on it. Docking each candidate
into PPIL4's (unrelated, enzymatic) active site is a cheap second signal
for ranking, not a claim about the true ternary mechanism.

This is a fast screening pass (Vina only, no HADDOCK3) meant to rank many
candidates by combined (CRBN + PPIL4) predicted affinity. 07 then runs the
slow full ternary HADDOCK3 docking on only the top-ranked candidates --
see TOP_N there.
"""
import math
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PYTHON = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"


def _has_required_packages():
    try:
        import vina, meeko, rdkit  # noqa: F401
        return True
    except ImportError:
        return False


# This script needs vina/meeko/rdkit, which live in the SYSTEM python, not
# the .venv-haddock3 venv used by 05/07. If they're missing -- e.g. the
# wrong venv is active, or the IDE's Run button used a different interpreter
# -- relaunch under the system python automatically instead of failing deep
# inside prepare_receptor(). The sys.executable check guards against looping
# if the system python itself is ever missing these packages.
if not _has_required_packages() and sys.executable != SYSTEM_PYTHON:
    os.execv(SYSTEM_PYTHON, [SYSTEM_PYTHON] + sys.argv)

from Bio import Align
from Bio.Align import substitution_matrices

REFERENCE_PDB = os.path.join(SCRIPT_DIR, "CRBN-Thalidomide-SALL4_(Ryan).pdb")
CRBN_RECEPTOR_PDB = os.path.join(SCRIPT_DIR, "CRBN_receptor_thalidomide_Ryan.pdb")
PPIL4_SOURCE_PDB = os.path.join(SCRIPT_DIR, "PPIL4_alphafold_(Ryan).pdb")
OUT_DIR = os.path.join(SCRIPT_DIR, "docking_tmp", "haddock3_novel_candidate")
SCREENING_SUMMARY_TSV = os.path.join(OUT_DIR, "screening_summary.tsv")

# Same CRBN-glutarimide degron chemotype as thalidomide/lenalidomide/pomalidomide
# (scored P(CRBN-glue)=1.000 by the Step-3 RF classifier), each with a
# different candidate extension -- picked from crbn_glue_compounds_(Ryan).txt.
CANDIDATES = [
    ("novel_candidate_1", "O=C1CCC(N2Cc3cc(NC(=O)c4cn5cc(Cl)ccc5n4)ccc3C2=O)C(=O)N1"),
    # add more (name, SMILES) pairs here
]


def thalidomide_box(reference_pdb, padding=14, min_size=20):
    coords = []
    with open(reference_pdb) as f:
        for line in f:
            if line.startswith("HETATM") and line[17:20].strip() == "EF2" and line[21] == "A":
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    xs, ys, zs = zip(*coords)
    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    size_x = max(max(xs) - min(xs) + padding, min_size)
    size_y = max(max(ys) - min(ys) + padding, min_size)
    size_z = max(max(zs) - min(zs) + padding, min_size)
    return (cx, cy, cz), (size_x, size_y, size_z)


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


def residue_box(pdb_path, chain, residues, padding=10, min_size=20):
    residues = set(residues)
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain and int(line[22:26]) in residues:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    xs, ys, zs = zip(*coords)
    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    size_x = max(max(xs) - min(xs) + padding, min_size)
    size_y = max(max(ys) - min(ys) + padding, min_size)
    size_z = max(max(zs) - min(zs) + padding, min_size)
    return (cx, cy, cz), (size_x, size_y, size_z)


def prepare_receptor(pdb_path, out_base, center, size):
    # Invoke via `-m` (module path) rather than relying on the
    # mk_prepare_receptor.py console script being on PATH -- meeko installs
    # that script under ~/Library/Python/3.12/bin, which isn't always on
    # PATH depending on how this script gets launched (IDE run button vs.
    # a shell that's sourced ~/.zprofile).
    subprocess.run(
        [sys.executable, "-m", "meeko.cli.mk_prepare_receptor",
         "--read_pdb", pdb_path, "-o", out_base, "-p", "-v",
         "--box_center", str(center[0]), str(center[1]), str(center[2]),
         "--box_size", str(size[0]), str(size[1]), str(size[2])],
        check=True,
    )


def smiles_to_ligand_pdbqt(smiles, out_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid SMILES")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        raise ValueError("3D embedding failed")
    AllChem.MMFFOptimizeMolecule(mol)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
    with open(out_path, "w") as f:
        f.write(pdbqt_string)


def dock_ligand(receptor_pdbqt, ligand_pdbqt, center, size, out_poses_pdbqt, exhaustiveness=16):
    from vina import Vina

    v = Vina(sf_name="vina", verbosity=1)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=list(center), box_size=list(size))
    v.dock(exhaustiveness=exhaustiveness, n_poses=10)
    v.write_poses(out_poses_pdbqt, n_poses=1, overwrite=True)
    energies = v.energies(n_poses=1)
    return float(energies[0][0])


def pdbqt_pose_to_pdb_lines(pdbqt_path, chain="A", resname="LIG", resnum=900):
    """Extract the first MODEL's atoms from a Vina pose pdbqt as plain PDB ATOM/HETATM lines."""
    lines = []
    serial = 9000
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM") or line.startswith("HETATM"):
                name = line[12:16]
                x, y, z = line[30:38], line[38:46], line[46:54]
                occ_temp = line[54:66] if len(line) >= 66 else "  1.00  0.00"
                element = line[76:78] if len(line) >= 78 else f" {name.strip()[0]}"
                serial += 1
                pdb_line = (
                    f"HETATM{serial:5d} {name:<4s}{resname:>3s} {chain}{resnum:4d}    "
                    f"{x:>8s}{y:>8s}{z:>8s}{occ_temp}          {element.strip():>2s}\n"
                )
                lines.append(pdb_line)
    return lines


def merge_receptor_and_ligand(receptor_pdb, ligand_lines, out_pdb):
    with open(receptor_pdb) as f:
        receptor_lines = [l for l in f if l.startswith("ATOM") or l.startswith("HETATM")]
    with open(out_pdb, "w") as f:
        f.writelines(receptor_lines)
        f.writelines(ligand_lines)
        f.write("END\n")


def find_ligand_contacts(receptor_pdb, ligand_lines, cutoff=4.5):
    protein_atoms = []
    with open(receptor_pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                resnum = int(line[22:26])
                protein_atoms.append((resnum, float(line[30:38]), float(line[38:46]), float(line[46:54])))

    lig_coords = []
    for line in ligand_lines:
        lig_coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

    contacts = set()
    for resnum, x, y, z in protein_atoms:
        for lx, ly, lz in lig_coords:
            d2 = (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2
            if d2 <= cutoff ** 2:
                contacts.add(resnum)
                break
    return sorted(contacts)


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


def check_environment():
    """Fail fast with a clear message if run under the wrong interpreter.

    This script needs the SYSTEM python (vina/meeko/rdkit installed there),
    not the .venv-haddock3 venv used by 05/07 -- that venv is for the
    haddock3 CLI only and doesn't have these packages.
    """
    missing = []
    for module in ("vina", "meeko", "rdkit"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        sys.exit(
            f"Missing packages: {', '.join(missing)}. You're running this with:\n"
            f"  {sys.executable}\n"
            "This script needs the SYSTEM python, not the .venv-haddock3 venv. Run:\n"
            '  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 '
            '"06_dock_candidate_crbn_(Ryan).py"'
        )


def main():
    check_environment()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("== Computing CRBN docking box from reference thalidomide position ==")
    crbn_center, crbn_size = thalidomide_box(REFERENCE_PDB)
    print(f"Box center: {crbn_center}, size: {crbn_size}")

    crbn_receptor_base = os.path.join(OUT_DIR, "CRBN_receptor")
    print("== Preparing CRBN Vina receptor ==")
    prepare_receptor(CRBN_RECEPTOR_PDB, crbn_receptor_base, crbn_center, crbn_size)
    crbn_receptor_pdbqt = crbn_receptor_base + ".pdbqt"

    print("\n== Computing PPIL4 docking box from CypA-homology pocket residues ==")
    ppil4_active = ppil4_pocket_residues()
    print("PPIL4 pocket residues:", ppil4_active)
    ppil4_center, ppil4_size = residue_box(PPIL4_SOURCE_PDB, "A", ppil4_active)
    print(f"Box center: {ppil4_center}, size: {ppil4_size}")

    ppil4_receptor_base = os.path.join(OUT_DIR, "PPIL4_receptor")
    print("== Preparing PPIL4 Vina receptor ==")
    prepare_receptor(PPIL4_SOURCE_PDB, ppil4_receptor_base, ppil4_center, ppil4_size)
    ppil4_receptor_pdbqt = ppil4_receptor_base + ".pdbqt"

    results = []
    for candidate_name, candidate_smiles in CANDIDATES:
        print(f"\n== Candidate: {candidate_name} ==")
        candidate_dir = os.path.join(OUT_DIR, candidate_name)
        os.makedirs(candidate_dir, exist_ok=True)

        print(f"== Preparing ligand: {candidate_name} ==")
        ligand_pdbqt = os.path.join(candidate_dir, f"{candidate_name}.pdbqt")
        smiles_to_ligand_pdbqt(candidate_smiles, ligand_pdbqt)

        print("== Docking against CRBN with Vina ==")
        crbn_poses_pdbqt = os.path.join(candidate_dir, f"{candidate_name}_crbn_poses.pdbqt")
        crbn_affinity = dock_ligand(crbn_receptor_pdbqt, ligand_pdbqt, crbn_center, crbn_size, crbn_poses_pdbqt)
        print(f"CRBN affinity: {crbn_affinity:.2f} kcal/mol")

        print("== Docking against PPIL4 with Vina ==")
        ppil4_poses_pdbqt = os.path.join(candidate_dir, f"{candidate_name}_ppil4_poses.pdbqt")
        ppil4_affinity = dock_ligand(ppil4_receptor_pdbqt, ligand_pdbqt, ppil4_center, ppil4_size, ppil4_poses_pdbqt)
        print(f"PPIL4 affinity: {ppil4_affinity:.2f} kcal/mol")

        print("== Extracting top CRBN pose and merging with CRBN receptor ==")
        ligand_lines = pdbqt_pose_to_pdb_lines(crbn_poses_pdbqt)
        merged_pdb = os.path.join(candidate_dir, "CRBN_candidate_complex.pdb")
        merge_receptor_and_ligand(CRBN_RECEPTOR_PDB, ligand_lines, merged_pdb)
        print(f"Wrote {merged_pdb}")

        print("== Finding CRBN contact residues ==")
        contacts = find_ligand_contacts(CRBN_RECEPTOR_PDB, ligand_lines)
        print("CRBN active (candidate-contact) residues:", contacts)

        contacts_path = os.path.join(candidate_dir, "crbn_contacts.txt")
        with open(contacts_path, "w") as f:
            f.write(" ".join(str(r) for r in contacts) + "\n")
            f.write(f"{crbn_affinity}\n")
        print(f"Wrote {contacts_path}")

        combined_affinity = crbn_affinity + ppil4_affinity
        results.append({
            "name": candidate_name, "crbn_affinity": crbn_affinity,
            "ppil4_affinity": ppil4_affinity, "combined_affinity": combined_affinity,
            "n_contacts": len(contacts),
        })

    results.sort(key=lambda r: r["combined_affinity"])
    print_table(
        results,
        [("name", "name"),
         ("crbn (kcal/mol)", lambda r: f"{r['crbn_affinity']:.2f}"),
         ("ppil4 (kcal/mol)", lambda r: f"{r['ppil4_affinity']:.2f}"),
         ("combined", lambda r: f"{r['combined_affinity']:.2f}"),
         ("n_contacts", lambda r: str(r["n_contacts"]))],
        title="Vina screening results (best combined affinity first)",
    )

    with open(SCREENING_SUMMARY_TSV, "w") as f:
        f.write("name\tcrbn_affinity\tppil4_affinity\tcombined_affinity\n")
        for r in results:
            f.write(f"{r['name']}\t{r['crbn_affinity']}\t{r['ppil4_affinity']}\t{r['combined_affinity']}\n")
    print(f"\nWrote {SCREENING_SUMMARY_TSV}")


if __name__ == "__main__":
    main()
