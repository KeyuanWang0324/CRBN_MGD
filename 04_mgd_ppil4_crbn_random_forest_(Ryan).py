# -*- coding: utf-8 -*-
"""
CRBN Glue-Chemotype Classifier -- Random Forest, 80/20 Split with LOOCV
=========================================================================
Trains on the pooled CRBN / CRBN-neosubstrate-ternary-complex dataset built
by 03_mgd_ppil4_crbn_screening.py (686 unique compounds, deduped by
canonical SMILES so no compound can leak across train/test or across LOOCV
folds).

Approach (same structure as the original CRBN binder script):
  1. 80/20 train/test split (test set untouched until final evaluation).
  2. LOOCV within the 80% training set for a robust internal AUC estimate.
  3. Train a final Random Forest on the full 80% training set.
  4. Evaluate once on the untouched 20% test set.

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

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

RDLogger.DisableLog("rdApp.*")

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


def run_loocv(X_train, y_train):
    loo = LeaveOneOut()
    n = X_train.shape[0]

    y_true = np.zeros(n, dtype=int)
    y_pred = np.zeros(n, dtype=int)
    y_proba = np.zeros(n, dtype=float)

    for i, (tr_idx, te_idx) in enumerate(loo.split(X_train)):
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_train[tr_idx], y_train[tr_idx])

        y_true[te_idx[0]] = y_train[te_idx[0]]
        y_proba[te_idx[0]] = clf.predict_proba(X_train[te_idx])[:, 1][0]
        y_pred[te_idx[0]] = clf.predict(X_train[te_idx])[0]

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  LOOCV progress: {i + 1}/{n}")

    return y_true, y_pred, y_proba


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

    print(f"Running LOOCV on {X_train.shape[0]} training samples...")
    loo_true, loo_pred, loo_proba = run_loocv(X_train, y_train)
    loo_auc = evaluate(loo_true, loo_pred, loo_proba, label="Training LOOCV")
    plot_roc(
        loo_true, loo_proba, loo_auc,
        title="ROC Curve -- CRBN Glue Chemotype (Training Set LOOCV)",
        filename="crbn_glue_roc_curve_train_loocv_(Ryan).png",
    )

    final_clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )
    final_clf.fit(X_train, y_train)

    test_pred = final_clf.predict(X_test)
    test_proba = final_clf.predict_proba(X_test)[:, 1]
    test_auc = evaluate(y_test, test_pred, test_proba, label="Held-out Test Set")
    plot_roc(
        y_test, test_proba, test_auc,
        title="ROC Curve -- CRBN Glue Chemotype (Held-out 20% Test Set)",
        filename="crbn_glue_roc_curve_test_(Ryan).png",
    )

    print("\n=== Summary ===")
    print(f"Training LOOCV AUC: {loo_auc:.4f}")
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
    scored = score_candidates(final_clf, list(example_candidates.values()))
    for (name, _), (smi, proba) in zip(example_candidates.items(), scored):
        p_str = f"{proba:.3f}" if proba is not None else "invalid SMILES"
        print(f"  {name}: P(CRBN-glue-competent) = {p_str}")
