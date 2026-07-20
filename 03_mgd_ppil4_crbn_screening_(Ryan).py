# -*- coding: utf-8 -*-
"""
CRBN Molecular-Glue-Degrader (MGD) Screening Model -- for PPIL4 MGD design
============================================================================
Goal: build a Random Forest classifier that predicts whether a small
molecule has a "CRBN-glue-competent" chemotype -- i.e. the chemistry
needed for the CRBN-recruiting half of a molecular glue degrader (MGD).

Why this, and not a direct "PPIL4 degrader" model
---------------------------------------------------
ChEMBL has essentially no PPIL4 bioactivity data usable for ML: as of this
pull, PPIL4 (CHEMBL5725153) has only 6 activity records, all from a single
compound (Molibresib) in a chemoproteomic dose-response panel. There is no
known CRBN-PPIL4 molecular glue in any public database -- that's exactly
the novel hypothesis this project is testing, so by definition no training
data for "degrades PPIL4" can exist yet.

What *does* exist: ChEMBL curates several CRBN/neosubstrate ternary-complex
targets (CRBN-CyclinK, CRBN-BCL2, CRBN-HDAC4, ...) in addition to the plain
CRBN single-protein target. These come from real molecular glue degrader
programs (e.g. CDK12/CyclinK glues, BCL2 glues) and are labeled with
degradation-relevant readouts (DC50, Emax, EC50), not just simple binding
affinity. Pooling these gives a much better proxy dataset for "CRBN-glue
pharmacophore" than plain CRBN-binder data alone.

This model answers: "does this candidate molecule have the chemistry of a
compound capable of engaging CRBN in a glue-type ternary complex?" It is a
triage filter for the CRBN-recruiting warhead/chemotype of a prospective
PPIL4-targeting MGD -- NOT a prediction that a given molecule will degrade
PPIL4. Actually identifying CRBN-PPIL4 glues requires the structure-based
ternary-complex modeling (CRBN surface + PPIL4 surface complementarity)
described in the project's computational phase; this RF (chemical
fingerprint only, no target structure) is complementary to that, not a
replacement for it. Use it to (a) pre-filter virtual libraries down to
CRBN-glue-plausible chemotypes before expensive docking, and/or (b)
sanity-check structure-based PPIL4 leads for CRBN-compatible chemistry
before committing to wet-lab synthesis.

Outputs:
    X : np.ndarray, shape (n_unique_compounds, 2048)  -- Morgan fingerprints
    Y : np.ndarray, shape (n_unique_compounds,)        -- glue-competent label (0/1)
Saved to disk as X_crbn_glue_fingerprints.npy / Y_crbn_glue_labels.npy
"""

import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2

# Potency/effect thresholds for calling a record "active" (glue/binder-competent)
POTENCY_THRESHOLD_NM = 10000   # DC50 / EC50 / IC50 / Ki / Kd <= 10 uM -> active
EFFECT_THRESHOLD_PCT = 50.0    # Emax / Activity / Inhibition (%) >= 50% -> active

# CRBN and CRBN/neosubstrate ternary-complex targets with usable record counts
# (identified by querying ChEMBL target/search for "CRBN" / "cereblon" and
# checking activity counts per target; targets with <10 records dropped as
# single-compound chemoproteomic panel noise, not real SAR).
TARGET_IDS = {
    "CHEMBL3763008": "CRBN (single protein)",
    "CHEMBL4523685": "CRBN / BCL2 (glue ternary complex)",
    "CHEMBL4296102": "CRBN / Casein kinase I alpha (glue ternary complex)",
    "CHEMBL4296127": "CRBN / HDAC4 (glue ternary complex)",
}


def fetch_all_activities(target_chembl_id, batch_size=1000):
    url = f"{CHEMBL_BASE_URL}/activity"
    all_records = []
    offset = 0
    while True:
        params = {
            "target_chembl_id": target_chembl_id,
            "format": "json",
            "limit": batch_size,
            "offset": offset,
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        activities = data.get("activities", [])
        if not activities:
            break
        all_records.extend(activities)

        total = data["page_meta"]["total_count"]
        offset += batch_size
        if offset >= total:
            break
    return all_records


def label_record(record, potency_threshold_nm=POTENCY_THRESHOLD_NM,
                  effect_threshold_pct=EFFECT_THRESHOLD_PCT):
    """
    Label a single ChEMBL activity record as glue/binder-active (1),
    inactive (0), or unlabelable (None).

    Priority order (most to least informative for glue pharmacology):
      1. Degradation/binding potency in nM (DC50, EC50, IC50, Ki, Kd)
      2. Percent effect (Emax, Activity, Inhibition) at reported dose
      3. Curated activity_comment text ("active"/"inactive")
    """
    std_type = (record.get("standard_type") or "").upper()
    std_value = record.get("standard_value")
    std_units = (record.get("standard_units") or "").lower()

    potency_types = {"DC50", "EC50", "EC90", "IC50", "KI", "KD"}
    if std_type in potency_types and std_value is not None and std_units == "nm":
        try:
            value = float(std_value)
        except (TypeError, ValueError):
            return None
        return 1 if value <= potency_threshold_nm else 0

    percent_types = {"EMAX", "ACTIVITY", "INHIBITION"}
    if std_type in percent_types and std_value is not None and std_units in ("%", ""):
        try:
            value = float(std_value)
        except (TypeError, ValueError):
            return None
        return 1 if value >= effect_threshold_pct else 0

    comment = (record.get("activity_comment") or "").lower()
    if "active" in comment and "inactive" not in comment:
        return 1
    if "inactive" in comment:
        return 0

    return None


def smiles_to_fingerprint(smiles, n_bits=FINGERPRINT_BITS, radius=FINGERPRINT_RADIUS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_dataset():
    """
    Pull activities from every target in TARGET_IDS, dedupe by canonical
    SMILES (majority-vote the label across duplicate/repeat records so the
    same compound can't land in both train and test with conflicting or
    redundant rows), and return (compounds, X, Y).
    """
    from collections import defaultdict

    label_votes = defaultdict(list)  # canonical_smiles -> [label, label, ...]

    for target_id, description in TARGET_IDS.items():
        print(f"Fetching {target_id} ({description}) ...")
        records = fetch_all_activities(target_id)
        print(f"  {len(records)} raw activity records")

        kept = 0
        for record in records:
            smiles = record.get("canonical_smiles")
            if not smiles:
                continue
            label = label_record(record)
            if label is None:
                continue
            label_votes[smiles].append(label)
            kept += 1
        print(f"  {kept} labelable records")

    print(f"\nUnique compounds across all targets: {len(label_votes)}")

    compounds, X_rows, Y_labels = [], [], []
    skipped_bad_smiles = 0
    for smiles, votes in label_votes.items():
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            skipped_bad_smiles += 1
            continue
        # majority vote across all records for this compound; ties -> active
        # (glue chemotypes are rare positives, so ties lean toward keeping them)
        label = 1 if sum(votes) >= len(votes) / 2 else 0
        compounds.append(smiles)
        X_rows.append(fp)
        Y_labels.append(label)

    X = np.array(X_rows)
    Y = np.array(Y_labels)

    print(f"Skipped (invalid SMILES): {skipped_bad_smiles}")
    print(f"\nFinal dataset: X {X.shape}, Y {Y.shape}")
    print(f"Glue/binder-active (Y=1): {int(Y.sum())}")
    print(f"Inactive (Y=0): {int((Y == 0).sum())}")

    return compounds, X, Y


if __name__ == "__main__":
    compounds, X, Y = build_dataset()
    np.save("X_crbn_glue_fingerprints_(Ryan).npy", X)
    np.save("Y_crbn_glue_labels_(Ryan).npy", Y)
    with open("crbn_glue_compounds_(Ryan).txt", "w") as f:
        f.write("\n".join(compounds))
    print("\nSaved X_crbn_glue_fingerprints_(Ryan).npy, Y_crbn_glue_labels_(Ryan).npy, crbn_glue_compounds_(Ryan).txt")
