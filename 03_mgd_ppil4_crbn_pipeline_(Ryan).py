# -*- coding: utf-8 -*-
"""
CRBN Molecular-Glue-Degrader (MGD) Screening Model for PPIL4
================================================================
Step 1 is the original CRBN single-protein binder pipeline + baseline
Random Forest. Steps 2-3 extend it: pool ChEMBL CRBN single-protein and
CRBN/neosubstrate ternary-complex bioactivity data (CyclinK, BCL2, HDAC4),
dedupe by compound, and train a Random Forest to predict CRBN-glue-
competent chemotype -- a pre-filter for candidate PPIL4-targeting MGD
warheads. Step 4 combines that classifier with PPIL4 docking into a
composite MGD-likelihood scorer. See each step's docstring for caveats:
this predicts CRBN-glue chemistry plausibility, not confirmed PPIL4
degradation (no such data exists publicly yet).

Setup (install once):
    pip install rdkit scikit-learn vina meeko gemmi biopython

Running this script top to bottom re-runs every step, including the
ChEMBL pulls and PPIL4 docking -- which is slow and pulls fresh data.
Steps 1-3 also overwrite the .npy/.png outputs already checked into this
directory. If you only want to score candidate SMILES against the
existing trained classifier + receptor, comment out the Step 1-3
`if __name__ == "__main__":` blocks below and just run Step 4.
"""

import argparse
import contextlib
import math
import os
import sys
import warnings

import numpy as np


# =============================================================================
# Step 1: CRBN Bioactivity Pipeline + Baseline RF Classifier (original)
# =============================================================================

"""
CRBN (Cereblon) Bioactivity Pipeline
=====================================
Pulls all bioactivity records for the CRBN target from ChEMBL (EMBL-EBI),
converts each compound's SMILES into a 2048-bit Morgan fingerprint, and
labels each record as a binder (1) or non-binder (0) based on a
Ki/IC50/EC50/Kd potency threshold.

Outputs:
    X : np.ndarray, shape (n_valid_records, 2048)  -- fingerprint matrix
    Y : np.ndarray, shape (n_valid_records,)        -- binder labels (0/1)

Saved to disk as X_crbn_fingerprints.npy and Y_crbn_binders.npy
"""

import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit SMILES parsing warnings

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
ACTIVITY_THRESHOLD_NM = 10000  # binder cutoff in nanomolar (adjust as needed)
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2


# ---------------------------------------------------------------------------
# Step 1: Find the CRBN target in ChEMBL
# ---------------------------------------------------------------------------
def get_crbn_target_id():
    """
    Search ChEMBL (EMBL-EBI) for the CRBN (Cereblon) target and return
    its ChEMBL target ID and preferred name.
    """
    url = f"{CHEMBL_BASE_URL}/target/search"
    params = {"q": "CRBN", "format": "json"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    targets = data.get("targets", [])
    if not targets:
        raise ValueError("No CRBN target found in ChEMBL")

    # Prefer a single-protein human target
    for t in targets:
        if t.get("target_type") == "SINGLE PROTEIN" and t.get("organism") == "Homo sapiens":
            return t["target_chembl_id"], t["pref_name"]

    # fallback: first hit
    return targets[0]["target_chembl_id"], targets[0]["pref_name"]


# ---------------------------------------------------------------------------
# Step 2: Pull every bioactivity record against CRBN
# ---------------------------------------------------------------------------
def fetch_all_activities(target_chembl_id, batch_size=1000):
    """
    Page through ChEMBL's /activity endpoint to retrieve all bioactivity
    records for the given target (e.g. ~1432 records for CRBN).
    """
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
        resp = requests.get(url, params=params)
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


# ---------------------------------------------------------------------------
# Step 3: Label each record as binder / non-binder
# ---------------------------------------------------------------------------
def label_binder(record, threshold_nm=ACTIVITY_THRESHOLD_NM):
    """
    Determine binder (1) / non-binder (0) / unknown (None) from a single
    ChEMBL activity record.

    Applies the same threshold to Ki, IC50, EC50, and Kd values reported
    in nanomolar units. Falls back to ChEMBL's curated activity_comment
    field ("active"/"inactive") when no usable numeric value is present.

    NOTE: Ki/Kd are direct binding affinity measurements, while IC50/EC50
    are assay-dependent functional readouts -- they are not strictly
    equivalent. This function currently treats them the same; consider
    restricting to Ki/Kd only if you want a stricter "binder" definition.
    """
    std_type = (record.get("standard_type") or "").upper()
    std_value = record.get("standard_value")
    std_units = (record.get("standard_units") or "").lower()

    binding_types = {"KI", "IC50", "EC50", "KD"}

    if std_type in binding_types and std_value is not None and std_units == "nm":
        try:
            value = float(std_value)
        except (TypeError, ValueError):
            return None
        return 1 if value <= threshold_nm else 0

    # fallback: curated text label
    comment = (record.get("activity_comment") or "").lower()
    if "active" in comment and "inactive" not in comment:
        return 1
    if "inactive" in comment:
        return 0

    return None  # not enough info to label


# ---------------------------------------------------------------------------
# Step 4: SMILES -> molecule -> molecular weight -> fingerprint
# ---------------------------------------------------------------------------
def process_smiles(smiles, n_bits=FINGERPRINT_BITS, radius=FINGERPRINT_RADIUS):
    """
    Convert a SMILES string into:
      - an RDKit molecule
      - molecular weight (stored as a property on the molecule)
      - a fingerprint (n_bits-length numpy array of 0s/1s)

    Returns (mol, mol_weight, fingerprint_array), or (None, None, None)
    if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    from rdkit.Chem import Descriptors
    mol_weight = Descriptors.MolWt(mol)
    mol.SetDoubleProp("molecular_weight", mol_weight)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    fp_array = np.zeros((n_bits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_array)

    return mol, mol_weight, fp_array


def smiles_to_fingerprint(smiles, n_bits=FINGERPRINT_BITS, radius=FINGERPRINT_RADIUS):
    """Lightweight version used when only the fingerprint (not MW) is needed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ---------------------------------------------------------------------------
# Main pipeline: build X (fingerprints) and Y (binder labels)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- find CRBN target ---
    target_id, target_name = get_crbn_target_id()
    print(f"Target: {target_name} ({target_id})")

    # --- pull all bioactivity records ---
    records = fetch_all_activities(target_id)
    print(f"Total activity records pulled: {len(records)}")

    X_rows = []
    Y_labels = []
    skipped_no_smiles = 0
    skipped_no_label = 0
    skipped_bad_smiles = 0

    for record in records:
        smiles = record.get("canonical_smiles")
        if not smiles:
            skipped_no_smiles += 1
            continue

        label = label_binder(record)
        if label is None:
            skipped_no_label += 1
            continue

        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            skipped_bad_smiles += 1
            continue

        X_rows.append(fp)
        Y_labels.append(label)

    X = np.array(X_rows)    # shape: (n_valid_records, 2048)
    Y = np.array(Y_labels)  # shape: (n_valid_records,)

    print(f"\nSkipped (no SMILES): {skipped_no_smiles}")
    print(f"Skipped (no activity label): {skipped_no_label}")
    print(f"Skipped (invalid SMILES): {skipped_bad_smiles}")

    print(f"\nX shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Binders (Y=1): {int(Y.sum())}")
    print(f"Non-binders (Y=0): {int((Y == 0).sum())}")

    # --- save for downstream ML use ---
    np.save("X_crbn_fingerprints.npy", X)
    np.save("Y_crbn_binders.npy", Y)
    print("\nSaved X_crbn_fingerprints.npy and Y_crbn_binders.npy")

"""
CRBN Binder Classifier -- Random Forest, 80/20 Held-Out Split
==================================================================================
Approach:
  1. Split data: 80% train / 20% test (held out, untouched until the end).
  2. Train a Random Forest on the full 80% training set.
  3. Evaluate that model once on the untouched 20% test set (ROC/AUC on
     true held-out data).

(Earlier version also ran Leave-One-Out CV within the training set --
dropped because it trains one Random Forest per training sample, which
was too slow to be worth it; the held-out test set alone already gives an
unbiased AUC estimate.)
"""

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

RANDOM_STATE = 42
TEST_SIZE = 0.20        # 20% held out as final test set
N_ESTIMATORS = 200


def load_data():
    X = np.load("X_crbn_fingerprints.npy")
    Y = np.load("Y_crbn_binders.npy")
    return X, Y


def evaluate(y_true, y_pred, y_proba, label=""):
    print(f"\nClassification report ({label}):")
    print(classification_report(y_true, y_pred, target_names=["Non-binder", "Binder"]))

    print(f"Confusion matrix ({label}):")
    print(confusion_matrix(y_true, y_pred))

    auc_score = roc_auc_score(y_true, y_proba)
    print(f"\n{label} AUC score: {auc_score:.4f}")

    return auc_score


def plot_roc(y_true, y_proba, auc_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved ROC curve to {filename}")


if __name__ == "__main__":
    X, Y = load_data()
    print(f"Loaded X: {X.shape}, Y: {Y.shape}\n")

    # --- Step 1: 80/20 split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y
    )
    print(f"Train set: {X_train.shape[0]} samples ({1 - TEST_SIZE:.0%})")
    print(f"Test set:  {X_test.shape[0]} samples ({TEST_SIZE:.0%})\n")

    # --- Step 2: train final model on full training set ---
    # RF #1: CRBN BINDER classifier -- predicts binder (1) vs non-binder (0)
    # against plain CRBN, from raw ChEMBL binding data. Not glue-specific.
    crbn_binder_clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    crbn_binder_clf.fit(X_train, y_train)

    # --- Step 3: evaluate final model on untouched 20% test set ---
    test_pred = crbn_binder_clf.predict(X_test)
    test_proba = crbn_binder_clf.predict_proba(X_test)[:, 1]
    test_auc = evaluate(y_test, test_pred, test_proba, label="Held-out Test Set")
    plot_roc(
        y_test, test_proba, test_auc,
        title="ROC Curve -- CRBN Binder (Held-out 20% Test Set)",
        filename="crbn_roc_curve_test.png"
    )

    print("\n=== Summary ===")
    print(f"Held-out Test AUC:  {test_auc:.4f}")


# =============================================================================
# Step 2: Pooled CRBN Molecular-Glue Data Pipeline (ChEMBL)
# =============================================================================

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

RDLogger.DisableLog("rdApp.*")

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


# =============================================================================
# Step 3: CRBN Glue-Chemotype Random Forest Classifier
# =============================================================================

"""
CRBN Glue-Chemotype Classifier -- Random Forest, 80/20 Held-Out Split
=========================================================================
Trains on the pooled CRBN / CRBN-neosubstrate-ternary-complex dataset built
by Step 2 above (686 unique compounds, deduped by canonical SMILES so no
compound can leak across train/test).

Approach:
  1. 80/20 train/test split (test set untouched until final evaluation).
  2. Train a Random Forest on the full 80% training set.
  3. Evaluate once on the untouched 20% test set.

(Earlier version also ran Leave-One-Out CV within the training set for a
second internal estimate -- dropped because it trains one Random Forest
per training sample, which was too slow to be worth it here; the held-out
test set alone already gives an unbiased AUC estimate.)

Then: load a list of candidate SMILES (e.g. the ~20 structure-based PPIL4
leads from the docking phase, or known CRBN-glue chemotypes such as
glutarimide/isoindolinone scaffolds you're considering as the E3-recruiting
arm) and score each for predicted CRBN-glue-chemotype probability.

IMPORTANT CAVEAT: this model predicts "does this molecule look like known
CRBN-glue chemistry", not "will this molecule degrade PPIL4". No CRBN-PPIL4
ternary complex data exists publicly (that's the novel hypothesis this
project tests). Use the probability output as a chemistry-plausibility
filter alongside, not instead of, the structure-based CRBN-PPIL4 ternary
docking work.
"""

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 200
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2


def load_data():
    X = np.load("X_crbn_glue_fingerprints_(Ryan).npy")
    Y = np.load("Y_crbn_glue_labels_(Ryan).npy")
    return X, Y


def smiles_to_fingerprint(smiles, n_bits=FINGERPRINT_BITS, radius=FINGERPRINT_RADIUS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def evaluate(y_true, y_pred, y_proba, label=""):
    print(f"\nClassification report ({label}):")
    print(classification_report(y_true, y_pred, target_names=["Non-glue", "Glue-competent"]))

    print(f"Confusion matrix ({label}):")
    print(confusion_matrix(y_true, y_pred))

    auc_score = roc_auc_score(y_true, y_proba)
    print(f"\n{label} AUC score: {auc_score:.4f}")

    return auc_score


def plot_roc(y_true, y_proba, auc_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved ROC curve to {filename}")


def score_candidates(clf, smiles_list):
    """
    Score a list of candidate SMILES for predicted CRBN-glue-chemotype
    probability. Returns list of (smiles, probability_or_None).
    None means the SMILES failed to parse.
    """
    results = []
    for smi in smiles_list:
        fp = smiles_to_fingerprint(smi)
        if fp is None:
            results.append((smi, None))
            continue
        proba = clf.predict_proba(fp.reshape(1, -1))[:, 1][0]
        results.append((smi, proba))
    return results


if __name__ == "__main__":
    X, Y = load_data()
    print(f"Loaded X: {X.shape}, Y: {Y.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y
    )
    print(f"Train set: {X_train.shape[0]} samples ({1 - TEST_SIZE:.0%})")
    print(f"Test set:  {X_test.shape[0]} samples ({TEST_SIZE:.0%})\n")

    # RF #2: CRBN-GLUE CHEMOTYPE classifier -- predicts glue-competent (1) vs
    # not (0) from the pooled CRBN + CRBN/neosubstrate ternary-complex data.
    # This is the "does it have glue-like chemistry" filter, distinct from
    # RF #1's plain binder/non-binder call above.
    crbn_glue_clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )
    crbn_glue_clf.fit(X_train, y_train)

    test_pred = crbn_glue_clf.predict(X_test)
    test_proba = crbn_glue_clf.predict_proba(X_test)[:, 1]
    test_auc = evaluate(y_test, test_pred, test_proba, label="Held-out Test Set")
    plot_roc(
        y_test, test_proba, test_auc,
        title="ROC Curve -- CRBN Glue Chemotype (Held-out 20% Test Set)",
        filename="crbn_glue_roc_curve_test_(Ryan).png",
    )

    print("\n=== Summary ===")
    print(f"Held-out Test AUC:  {test_auc:.4f}")

    # --- Example: score candidate PPIL4-MGD warheads for CRBN-glue chemotype ---
    # Replace with your actual structure-based PPIL4 lead SMILES once available.
    # Included here are well-known CRBN-glue degron scaffolds (thalidomide,
    # lenalidomide, pomalidomide) as a sanity check -- they should score high.
    example_candidates = {
        "Thalidomide": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
        "Lenalidomide": "NC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2",
        "Pomalidomide": "NC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2=O",
    }
    print("\n=== Example candidate scoring (sanity check: known CRBN glues) ===")
    scored = score_candidates(crbn_glue_clf, list(example_candidates.values()))
    for (name, _), (smi, proba) in zip(example_candidates.items(), scored):
        p_str = f"{proba:.3f}" if proba is not None else "invalid SMILES"
        print(f"  {name}: P(CRBN-glue-competent) = {p_str}")


# =============================================================================
# Step 4: PPIL4 Docking + CRBN-PPIL4 MGD Composite Scorer
#
# Setup notes (install once):
#     pip install vina meeko gemmi rdkit biopython
#     brew install boost   (vina's compiled dependency)
#
# Local (macOS): Vina has no macOS wheel on PyPI, so it builds from source
# and needs Boost + SWIG. Homebrew's current Boost (1.90) also needs a
# newer C++ standard than Vina's setup.py hardcodes:
#     brew install boost swig
#     # patch vina's sdist: change "-std=c++11" -> "-std=c++17" in setup.py
#     # (pip install vina alone will fail on this machine's Boost otherwise)
#     CONDA_DEFAULT_ENV=x CONDA_PREFIX=/opt/homebrew pip3 install vina
# The CONDA_* env vars aren't about conda -- vina's setup.py only searches
# conda-env paths, /usr/local/include, or /usr/include for Boost, and Apple
# Silicon Homebrew lives at /opt/homebrew, so this fakes a "conda env" to
# point it there without needing sudo/symlinks into /usr/local.
#
# Colab (Linux x86_64): none of the above is needed -- Vina ships a
# prebuilt manylinux wheel there, so a plain `pip install vina` just works.
#
# Running this section with no changes prompts you to type in SMILES one at
# a time (blank line to finish) -- see the bottom of this file. Pass
# --sanity-check to score the known CRBN glues instead, or --smiles-file
# path.txt for batch scoring from a file.
# =============================================================================

"""
CRBN-PPIL4 Molecular-Glue-Degrader (MGD) Composite Scorer
=============================================================
Combines two independent scores into one "MGD likelihood" ranking for
candidate small molecules:

  1. CRBN-glue chemotype probability -- from the Random Forest trained in
     Step 3 above on pooled ChEMBL CRBN / CRBN-neosubstrate bioactivity
     data (ligand fingerprint only).

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

Usage:
    python3 "03_mgd_ppil4_crbn_pipeline_(Ryan).py"                        # type SMILES in interactively
    python3 "03_mgd_ppil4_crbn_pipeline_(Ryan).py" --sanity-check          # known CRBN glues (pipeline check)
    python3 "03_mgd_ppil4_crbn_pipeline_(Ryan).py" --smiles-file cands.txt # batch mode, one "SMILES,Name" per line
"""

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _suppress_native_stderr():
    """Vina's C++ core writes some warnings (e.g. 'At low exhaustiveness...')
    straight to the OS-level stderr (real fd 2), bypassing verbosity=0 and
    Python's own `warnings` module -- silence them at the file-descriptor
    level, only around the native call, so real Python exceptions still
    surface normally.
    """
    try:
        saved_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


from rdkit.Chem import rdFingerprintGenerator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECEPTOR_PDBQT = os.path.join(SCRIPT_DIR, "PPIL4_receptor_(Ryan).pdbqt")

# Docking box: centroid of PPIL4 catalytic-pocket residues homology-mapped
# from human CypA's active site (see build_ppil4_receptor() below for how
# this was derived). Precomputed once; re-derive if the receptor is rebuilt.
BOX_CENTER = (-2.050, 4.508, -19.059)
BOX_SIZE = (26.3, 31.6, 28.0)

# CRBN-glue RF classifier fingerprint settings (must match Steps 2/3 above)
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
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprintAsNumPy(mol).astype(int)


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

    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(RECEPTOR_PDBQT)
    v.set_ligand_from_file(ligand_pdbqt_path)
    v.compute_vina_maps(center=list(BOX_CENTER), box_size=list(BOX_SIZE))
    with _suppress_native_stderr():
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


# Known CRBN glues (should score high on CRBN-glue; PPIL4 docking score is
# uninformative for these -- included only as a pipeline sanity check).
SANITY_CHECK_CANDIDATES = [
    ("Thalidomide", "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"),
    ("Lenalidomide", "NC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2"),
    ("Pomalidomide", "NC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2=O"),
]


def get_manual_candidates():
    """Prompt for SMILES one at a time (optionally 'SMILES,Name'); blank line to finish."""
    print("Enter a SMILES per line (optionally 'SMILES,Name'). Blank line to finish.\n")
    candidates = []
    while True:
        line = input(f"  [{len(candidates) + 1}] SMILES: ").strip()
        if not line:
            break
        parts = line.split(",")
        smi = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else f"cand_{len(candidates) + 1}"
        candidates.append((name, smi))
    return candidates


def print_results_table(results):
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    ok.sort(key=lambda r: r["mgd_composite_score"], reverse=True)

    name_w = max([len(r["name"]) for r in results] + [4])
    header = (f"  {'Rank':<4} {'Name':<{name_w}} {'P(CRBN-glue)':>12} "
              f"{'PPIL4 kcal/mol':>15} {'P(PPIL4-bind)':>13} {'Composite':>10}")

    print("\n" + "=" * len(header))
    print("MGD Composite Scores (CRBN-glue x PPIL4-dock), best first")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, r in enumerate(ok, 1):
        print(f"  {i:<4} {r['name']:<{name_w}} {r['p_crbn_glue']:>12.3f} "
              f"{r['ppil4_affinity_kcal_mol']:>15.2f} {r['p_ppil4_bind']:>13.3f} "
              f"{r['mgd_composite_score']:>10.3f}")

    if failed:
        print("\n  Failed:")
        for r in failed:
            print(f"    {r['name']}: {r['error']}")
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", help="File of SMILES (optionally 'SMILES,Name' per line)")
    parser.add_argument("--sanity-check", action="store_true",
                         help="Score known CRBN glues (thalidomide/lenalidomide/pomalidomide) instead of prompting")
    parser.add_argument("--rebuild-receptor", action="store_true",
                         help="Re-fetch PPIL4 structure and rebuild the receptor PDBQT/box")
    args = parser.parse_args()

    if args.rebuild_receptor or not os.path.exists(RECEPTOR_PDBQT):
        build_ppil4_receptor()

    # Refit of RF #2 (the CRBN-glue chemotype classifier from Step 3 above) --
    # this is the P(CRBN-glue) half of the composite score below.
    X = np.load(os.path.join(SCRIPT_DIR, "X_crbn_glue_fingerprints_(Ryan).npy"))
    Y = np.load(os.path.join(SCRIPT_DIR, "Y_crbn_glue_labels_(Ryan).npy"))
    crbn_glue_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
    crbn_glue_clf.fit(X, Y)

    # --- everything above this point is setup; the interactive SMILES
    # --- prompt (when no --smiles-file / --sanity-check is passed) is the
    # --- last thing that runs in the whole script.
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
    elif args.sanity_check:
        candidates = SANITY_CHECK_CANDIDATES
    else:
        candidates = get_manual_candidates()

    if not candidates:
        print("No SMILES entered -- nothing to score.")
    else:
        results = score_candidates(candidates, crbn_glue_clf, tmp_dir=os.path.join(SCRIPT_DIR, "docking_tmp"))
        print_results_table(results)
