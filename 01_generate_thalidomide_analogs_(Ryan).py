"""
Generate thalidomide analogs by substituting groups on the phthalimide
benzo ring while keeping the glutarimide ring (the piperidine-2,6-dione
that binds CRBN's degron pocket) completely untouched.

Run with the SYSTEM python (has rdkit installed):
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
        "01_generate_thalidomide_analogs_(Ryan).py"

Writes 01_generated_analogs_(Ryan).csv (id, substituents, smiles) -- the raw
analog pool scored by 03 and 04 downstream -- and prints how many valid,
unique analog SMILES were generated.
"""
import csv
import os
import random
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PYTHON = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"


def _has_required_packages():
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


# rdkit lives in the SYSTEM python, not the .venv-haddock3 venv used by
# 05/07. Relaunch automatically if it's missing, regardless of which
# interpreter launched this script.
if not _has_required_packages() and sys.executable != SYSTEM_PYTHON:
    os.execv(SYSTEM_PYTHON, [SYSTEM_PYTHON] + sys.argv)

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit SMILES parsing warnings

THALIDOMIDE_SMILES = "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"
N_ANALOGS = 500
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "01_generated_analogs_(Ryan).csv")
RANDOM_SEED = 42

# Substituent fragments, each attached via a single bond in place of one
# ring hydrogen.
SUBSTITUENTS = {
    "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "NH2": "N", "OH": "O", "CH3": "C", "CH2CH3": "CC",
    "OCH3": "OC", "CN": "C#N", "NO2": "[N+](=O)[O-]",
    "CF3": "C(F)(F)F", "COOH": "C(=O)O", "CONH2": "C(=O)N",
    "SO2NH2": "S(=O)(=O)N", "NHCH3": "NC", "N(CH3)2": "N(C)C",
}


def benzo_ring_substitution_sites(mol):
    """Atom indices of the aromatic benzo-ring carbons that still carry a
    hydrogen -- valid substitution positions on 'the other ring', excluding
    the two ring-fusion carbons already bonded to the glutarimide side."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            return [i for i in ring if mol.GetAtomWithIdx(i).GetTotalNumHs() > 0]
    raise ValueError("Could not find the benzo ring in thalidomide's SMILES")


def substitute(base_mol, site_to_fragment):
    """Return a new mol with the given {ring_atom_idx: fragment_smiles}
    substituents attached in place of that atom's hydrogen, or None if the
    result doesn't sanitize (e.g. valence clash from a crowded position)."""
    rw = Chem.RWMol(base_mol)
    for site_idx, frag_smiles in site_to_fragment.items():
        frag = Chem.MolFromSmiles(frag_smiles)
        attach_idx = rw.GetNumAtoms()
        rw = Chem.RWMol(Chem.CombineMols(rw, frag))
        rw.AddBond(site_idx, attach_idx, Chem.BondType.SINGLE)
    try:
        mol_out = rw.GetMol()
        Chem.SanitizeMol(mol_out)
        return mol_out
    except Exception:
        return None


def main():
    random.seed(RANDOM_SEED)
    base = Chem.MolFromSmiles(THALIDOMIDE_SMILES)
    if base is None:
        raise ValueError("Invalid thalidomide SMILES")

    sites = benzo_ring_substitution_sites(base)
    print(f"Thalidomide: {Chem.MolToSmiles(base)}")
    print(f"Substitutable benzo-ring positions (glutarimide ring untouched): {sites}")
    print(f"Substituent groups available: {len(SUBSTITUENTS)}")

    seen = {Chem.MolToSmiles(base)}
    analogs = []
    attempts = 0
    max_attempts = N_ANALOGS * 200
    start = time.time()

    while len(analogs) < N_ANALOGS and attempts < max_attempts:
        attempts += 1
        n_subs = random.choice([1, 1, 2])  # mostly single substitution, sometimes two
        chosen_sites = random.sample(sites, k=min(n_subs, len(sites)))
        chosen = {site: random.choice(list(SUBSTITUENTS.items())) for site in chosen_sites}

        mol_out = substitute(base, {site: frag for site, (_, frag) in chosen.items()})
        if mol_out is None:
            continue
        smiles = Chem.MolToSmiles(mol_out)
        if smiles in seen:
            continue
        seen.add(smiles)
        analogs.append(smiles)
        if len(analogs) % 50 == 0:
            print(f"  ... {len(analogs)}/{N_ANALOGS} analogs generated "
                  f"({attempts} attempts, {time.time() - start:.0f}s elapsed)")

    if len(analogs) < N_ANALOGS:
        print(f"WARNING: only found {len(analogs)}/{N_ANALOGS} unique valid analogs "
              f"after {attempts} attempts.")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([[s] for s in analogs])

    print(f"Generated {len(analogs)} valid, unique analog SMILES ({attempts} attempts)")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
