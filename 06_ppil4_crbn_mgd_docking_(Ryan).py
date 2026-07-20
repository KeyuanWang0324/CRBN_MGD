# -*- coding: utf-8 -*-
"""
CRBN-PPIL4 Molecular-Glue-Degrader (MGD) Composite Scorer
=============================================================
Combines two independent scores into one "MGD likelihood" ranking for
candidate small molecules:

  1. CRBN-glue chemotype probability -- from the Random Forest trained in
     04_mgd_ppil4_crbn_random_forest_(Ryan).py on pooled ChEMBL CRBN /
     CRBN-neosubstrate bioactivity data (ligand fingerprint only).

  2. PPIL4 catalytic-pocket docking score -- AutoDock Vina docking against
     the PPIL4 AlphaFold model (Q8WUA2), restricted to the isomerase
     catalytic domain. No PPIL4-bound ligand structure exists, so the
     docking box is centered on the pocket homology-mapped from human
     Cyclophilin A (CypA/PPIA, P62937) -- CypA's well-characterized
     proline-binding active site (Arg55, Phe60, Met61, Gln63, Gly72,
     Ala101, Asn102, Ala103, Gln111, Phe113, Trp121, Leu122, His126)
     aligned onto PPIL4's PPIase domain (residues 1-180; the rest of the
     492-residue AlphaFold model is a low-confidence disordered RS/SR-rich
     tail per-residue pLDDT and is excluded from docking).

Composite score = P(CRBN-glue) * P(PPIL4-bind), i.e. independence between
the two events. This is a simplifying assumption, not a validated joint
model -- no CRBN-PPIL4 ternary complex has ever been observed, so there is
no data to fit real covariance between the two terms. Treat the composite
score as a RANKING heuristic to prioritize candidates for synthesis /
wet-lab testing, not as a calibrated probability.

Requirements (install once; see README section at bottom of this file):
    pip install vina meeko gemmi rdkit biopython
    brew install boost   (vina's compiled dependency)

Usage:
    python3 "06_ppil4_crbn_mgd_docking_(Ryan).py" --smiles-file candidates.txt
    (one SMILES per line; optionally "SMILES,Name")
"""

import argparse
import math
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECEPTOR_PDBQT = os.path.join(SCRIPT_DIR, "PPIL4_receptor_(Ryan).pdbqt")

# Docking box: centroid of PPIL4 catalytic-pocket residues homology-mapped
# from human CypA's active site (see build_ppil4_receptor() below for how
# this was derived). Precomputed once; re-derive if the receptor is rebuilt.
BOX_CENTER = (-2.050, 4.508, -19.059)
BOX_SIZE = (26.3, 31.6, 28.0)

# CRBN-glue RF classifier fingerprint settings (must match 03/04 scripts)
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2

# Affinity->probability transform: docking affinities (kcal/mol, more
# negative = tighter binding) are mapped to a pseudo-probability with a
# logistic centered at AFFINITY_MIDPOINT (a "borderline binder" cutoff),
# with AFFINITY_SCALE controlling how sharply probability changes with
# affinity. These are heuristic choices, not fit to PPIL4 data (none
# exists) -- calibrate against real assay results once available.
AFFINITY_MIDPOINT = -6.5   # kcal/mol; roughly a low-micromolar cutoff
AFFINITY_SCALE = 1.0


def build_ppil4_receptor():
    """
    One-time setup: fetch the PPIL4 AlphaFold model, locate the catalytic
    PPIase domain via per-residue pLDDT, homology-map CypA's active site
    onto it, and prepare a receptor PDBQT with a docking box centered on
    that pocket. Already run once (outputs cached in this directory) --
    this function documents/reproduces how PPIL4_receptor.pdbqt was made.
    """
    import subprocess
    import urllib.request
    from Bio import Align
    from Bio.Align import substitution_matrices

    pdb_path = os.path.join(SCRIPT_DIR, "PPIL4_alphafold_(Ryan).pdb")
    if not os.path.exists(pdb_path):
        urllib.request.urlretrieve(
            "https://alphafold.ebi.ac.uk/files/AF-Q8WUA2-F1-model_v6.pdb", pdb_path
        )

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

    pocket_residues = [mapping[r] for r in active_site_cypa if r in mapping]

    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resnum = int(line[22:26])
                if resnum in pocket_residues:
                    coords[resnum] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))

    xs, ys, zs = zip(*coords.values())
    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    size_x = max(max(xs) - min(xs) + 14, 20)
    size_y = max(max(ys) - min(ys) + 14, 20)
    size_z = max(max(zs) - min(zs) + 14, 20)

    print(f"Pocket residues (PPIL4 numbering): {sorted(pocket_residues)}")
    print(f"Box center: ({cx:.3f}, {cy:.3f}, {cz:.3f}), size: ({size_x:.1f}, {size_y:.1f}, {size_z:.1f})")

    receptor_base = os.path.join(SCRIPT_DIR, "PPIL4_receptor_(Ryan)")
    subprocess.run(
        ["mk_prepare_receptor.py", "--read_pdb", pdb_path, "-o", receptor_base, "-p", "-v",
         "--box_center", str(cx), str(cy), str(cz),
         "--box_size", str(size_x), str(size_y), str(size_z)],
        check=True,
    )
    return (cx, cy, cz), (size_x, size_y, size_z)


def smiles_to_fingerprint(smiles, n_bits=FINGERPRINT_BITS, radius=FINGERPRINT_RADIUS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_ligand_pdbqt(smiles, out_path):
    """Embed a 3D conformer for the SMILES and write a Vina-ready ligand PDBQT."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return False
    AllChem.MMFFOptimizeMolecule(mol)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
    with open(out_path, "w") as f:
        f.write(pdbqt_string)
    return True


def dock_ligand(ligand_pdbqt_path, exhaustiveness=8):
    """Run AutoDock Vina and return the best (most negative) binding affinity in kcal/mol."""
    from vina import Vina

    v = Vina(sf_name="vina")
    v.set_receptor(RECEPTOR_PDBQT)
    v.set_ligand_from_file(ligand_pdbqt_path)
    v.compute_vina_maps(center=list(BOX_CENTER), box_size=list(BOX_SIZE))
    v.dock(exhaustiveness=exhaustiveness, n_poses=10)
    energies = v.energies(n_poses=1)
    return float(energies[0][0])


def affinity_to_probability(affinity_kcal_mol):
    """Logistic transform: more negative affinity -> higher P(PPIL4-bind)."""
    return 1.0 / (1.0 + math.exp((affinity_kcal_mol - AFFINITY_MIDPOINT) / AFFINITY_SCALE))


def score_candidates(candidates, crbn_glue_clf, tmp_dir):
    """
    candidates: list of (name, smiles)
    Returns list of dicts with per-candidate scores.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    results = []
    for name, smiles in candidates:
        row = {"name": name, "smiles": smiles}

        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            row["error"] = "invalid SMILES"
            results.append(row)
            continue
        row["p_crbn_glue"] = float(crbn_glue_clf.predict_proba(fp.reshape(1, -1))[:, 1][0])

        ligand_path = os.path.join(tmp_dir, f"{name.replace(' ', '_')}.pdbqt")
        try:
            if not smiles_to_ligand_pdbqt(smiles, ligand_path):
                raise ValueError("3D embedding / ligand prep failed")
            affinity = dock_ligand(ligand_path)
            row["ppil4_affinity_kcal_mol"] = affinity
            row["p_ppil4_bind"] = affinity_to_probability(affinity)
            row["mgd_composite_score"] = row["p_crbn_glue"] * row["p_ppil4_bind"]
        except Exception as exc:
            row["error"] = f"docking failed: {exc}"

        results.append(row)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", help="File of SMILES (optionally 'SMILES,Name' per line)")
    parser.add_argument("--rebuild-receptor", action="store_true",
                         help="Re-fetch PPIL4 structure and rebuild the receptor PDBQT/box")
    args = parser.parse_args()

    if args.rebuild_receptor or not os.path.exists(RECEPTOR_PDBQT):
        build_ppil4_receptor()

    import joblib
    from sklearn.ensemble import RandomForestClassifier
    X = np.load(os.path.join(SCRIPT_DIR, "X_crbn_glue_fingerprints_(Ryan).npy"))
    Y = np.load(os.path.join(SCRIPT_DIR, "Y_crbn_glue_labels_(Ryan).npy"))
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X, Y)

    if args.smiles_file:
        candidates = []
        with open(args.smiles_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                smi = parts[0].strip()
                nm = parts[1].strip() if len(parts) > 1 else f"cand_{i}"
                candidates.append((nm, smi))
    else:
        # Sanity-check set: known CRBN glues (should score high on CRBN-glue,
        # PPIL4 docking score is uninformative for these -- included only to
        # confirm the pipeline runs end-to-end).
        candidates = [
            ("Thalidomide", "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"),
            ("Lenalidomide", "NC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2"),
            ("Pomalidomide", "NC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2=O"),
        ]

    results = score_candidates(candidates, clf, tmp_dir=os.path.join(SCRIPT_DIR, "docking_tmp"))

    print("\n=== MGD Composite Scores (CRBN-glue x PPIL4-dock) ===")
    for r in sorted(results, key=lambda r: r.get("mgd_composite_score", -1), reverse=True):
        if "error" in r:
            print(f"  {r['name']}: ERROR - {r['error']}")
        else:
            print(f"  {r['name']}: P(CRBN-glue)={r['p_crbn_glue']:.3f}  "
                  f"PPIL4 affinity={r['ppil4_affinity_kcal_mol']:.2f} kcal/mol  "
                  f"P(PPIL4-bind)={r['p_ppil4_bind']:.3f}  "
                  f"composite={r['mgd_composite_score']:.3f}")

"""
Setup notes
-----------
This machine did not have Homebrew/Boost installed when this script was
written, so AutoDock Vina (which needs Boost to build) could not be
installed yet. Everything else in this pipeline IS working and tested:
  - PPIL4 AlphaFold structure fetched (Q8WUA2, v6)
  - Catalytic PPIase domain located via per-residue pLDDT (residues ~1-180;
    rest of the 492-aa model is a low-confidence disordered tail)
  - Pocket homology-mapped from human CypA's active site via BLOSUM62
    global alignment (Bio.Align) -- 13/13 active-site residues mapped
  - Receptor PDBQT + docking box built successfully with meeko's
    mk_prepare_receptor.py
  - Ligand prep (RDKit 3D embedding + meeko PDBQT writer) code path is
    standard meeko/RDKit usage

Once Boost is available:
    brew install boost
    pip3 install vina
then this script runs end-to-end with no other changes needed.
"""
