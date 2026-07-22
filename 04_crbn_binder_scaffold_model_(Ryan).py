"""
Pull all ChEMBL bioactivity records for human CRBN (CHEMBL3763008), label
active/inactive by a 10 uM potency cutoff, dedupe by canonical SMILES
(majority vote across duplicate records), compute 2048-bit radius-2 Morgan
fingerprints, and train a RandomForest binder classifier.

Reports AUC / precision / recall under two evaluation regimes:
  1. Random 80/20 split -- optimistic, since structurally near-identical
     compounds (same scaffold, different R-group) can land on both sides.
  2. Scaffold (Bemis-Murcko) split -- all compounds sharing a scaffold are
     kept together in either train or test, so the model is evaluated on
     genuinely novel chemotypes it hasn't seen any analog of. This is the
     harder, more realistic generalization estimate.

Run with the SYSTEM python (has requests/rdkit/sklearn installed):
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
        "04_crbn_binder_scaffold_model_(Ryan).py"
"""
import os
import random
import sys
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PYTHON = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"


def _has_required_packages():
    try:
        import requests, rdkit, sklearn, numpy  # noqa: F401
        return True
    except ImportError:
        return False


# This script needs requests/rdkit/sklearn, missing from the .venv-haddock3
# venv used by 05/07. Relaunch under the system python automatically if
# they're not available, regardless of which interpreter launched this.
if not _has_required_packages() and sys.executable != SYSTEM_PYTHON:
    os.execv(SYSTEM_PYTHON, [SYSTEM_PYTHON] + sys.argv)

import numpy as np
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report

RDLogger.DisableLog("rdApp.*")

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
TARGET_CHEMBL_ID = "CHEMBL3763008"  # human cereblon (CRBN)
POTENCY_THRESHOLD_NM = 10000  # 10 uM cutoff for active/inactive
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 200

CANDIDATES_CSV = os.path.join(SCRIPT_DIR, "candidates.csv")
CANDIDATE_SCORES_CSV = os.path.join(SCRIPT_DIR, "crbn_binder_scores_(Ryan).csv")
MGD_SCORES_CSV = os.path.join(SCRIPT_DIR, "mgd_scores_(Ryan).csv")  # written by 03's Step 4
ACTIVE_CANDIDATES_CSV = os.path.join(SCRIPT_DIR, "active_candidates_(Ryan).csv")  # consumed by 06
ACTIVE_FRACTION = 0.20


def fetch_all_activities(target_chembl_id, batch_size=1000):
    """Page through ChEMBL's /activity endpoint for every record against this target."""
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
        print(f"  ... fetched {len(all_records)}/{total} activity records")
        offset += batch_size
        if offset >= total:
            break
    return all_records


def label_record(record, threshold_nm=POTENCY_THRESHOLD_NM):
    """Active (1) if a Ki/IC50/EC50/Kd potency is <= threshold_nm, inactive (0)
    if above, None if the record has no usable numeric value (skipped)."""
    std_type = (record.get("standard_type") or "").upper()
    std_value = record.get("standard_value")
    std_units = (record.get("standard_units") or "").lower()

    if std_type in {"KI", "IC50", "EC50", "KD"} and std_value is not None and std_units == "nm":
        try:
            value = float(std_value)
        except (TypeError, ValueError):
            return None
        return 1 if value <= threshold_nm else 0
    return None


def dedupe_majority_vote(records):
    """Group records by canonical SMILES, majority-vote the label across all
    records for that compound (ties -> active), and return (smiles_list,
    label_list) with one entry per unique compound."""
    votes_by_smiles = defaultdict(list)
    for record in records:
        smiles = record.get("canonical_smiles")
        if not smiles:
            continue
        label = label_record(record)
        if label is None:
            continue
        votes_by_smiles[smiles].append(label)

    smiles_list, label_list = [], []
    for smiles, votes in votes_by_smiles.items():
        smiles_list.append(smiles)
        label_list.append(1 if sum(votes) >= len(votes) / 2 else 0)
    return smiles_list, label_list


def smiles_to_fingerprint(smiles, n_bits=FINGERPRINT_BITS, radius=FINGERPRINT_RADIUS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_scaffold(smiles):
    """Bemis-Murcko scaffold SMILES for a molecule, or None if it can't be computed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def scaffold_split(smiles_list, test_size=TEST_SIZE, seed=RANDOM_STATE):
    """Split indices so every compound sharing a scaffold stays on the same
    side -- compounds with no computable scaffold get their own singleton
    group. Scaffold groups are shuffled, then greedily filled into the test
    set up to the target size."""
    groups_by_scaffold = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles) or f"__no_scaffold_{i}__"
        groups_by_scaffold[scaffold].append(i)

    groups = list(groups_by_scaffold.values())
    random.Random(seed).shuffle(groups)

    target_test = round(len(smiles_list) * test_size)
    test_idx, train_idx = [], []
    for group in groups:
        if len(test_idx) < target_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    print(f"Scaffold split: {len(groups)} unique scaffolds -> "
          f"{len(train_idx)} train compounds, {len(test_idx)} test compounds")
    return train_idx, test_idx


def train_and_evaluate(X_train, X_test, y_train, y_test, label):
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    )
    print(f"Training RandomForest ({N_ESTIMATORS} trees, {label}) ...")
    fit_start = time.time()
    clf.fit(X_train, y_train)
    print(f"  ... done in {time.time() - fit_start:.1f}s")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n=== {label} ===")
    print(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")
    print(classification_report(y_test, y_pred, target_names=["Non-binder", "Binder"], zero_division=0))
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    return {"auc": auc, "precision": precision, "recall": recall}


def score_candidates_file(clf, candidates_csv, out_csv):
    """Score every SMILES in candidates_csv (one per line, no header) with
    the given fitted classifier and write (name, smiles, p_crbn_binder,
    predicted_active) to out_csv."""
    import csv

    rows = []
    with open(candidates_csv) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i % 100 == 0:
            print(f"  ... scored {i}/{len(lines)} candidates")
        smiles = line.strip()
        if not smiles:
            continue
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            rows.append({"name": f"cand_{i}", "smiles": smiles, "p_crbn_binder": "", "predicted_active": ""})
            continue
        proba = float(clf.predict_proba(fp.reshape(1, -1))[:, 1][0])
        rows.append({
            "name": f"cand_{i}", "smiles": smiles,
            "p_crbn_binder": proba, "predicted_active": int(proba >= 0.5),
        })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "smiles", "p_crbn_binder", "predicted_active"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    return rows


def build_active_candidates(binder_rows, mgd_scores_csv, out_csv, active_fraction=ACTIVE_FRACTION):
    """Combine this script's P(CRBN-binder) ranking with 03's mgd_composite_score
    ranking (rank-sum aggregation -- lower combined rank is better) and write
    the top `active_fraction` of candidates to out_csv for use as 06's
    candidate list. Neither ranking alone decides "active"; a candidate has
    to rank well on both the plain-binder classifier (04) and the
    CRBN-glue-chemotype x PPIL4-dock composite score (03)."""
    import csv

    mgd_by_smiles = {}
    with open(mgd_scores_csv) as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            mgd_by_smiles[row["smiles"]] = row

    combined = []
    for row in binder_rows:
        if row["p_crbn_binder"] == "":
            continue
        mgd_row = mgd_by_smiles.get(row["smiles"])
        if mgd_row is None:
            continue
        combined.append({
            "name": row["name"], "smiles": row["smiles"],
            "p_crbn_binder": float(row["p_crbn_binder"]),
            "p_crbn_glue": float(mgd_row["p_crbn_glue"]),
            "p_ppil4_bind": float(mgd_row["p_ppil4_bind"]),
            "mgd_composite_score": float(mgd_row["mgd_composite_score"]),
        })

    if not combined:
        print(f"No candidates present in both {CANDIDATE_SCORES_CSV} and {mgd_scores_csv} "
              "-- skipping active-candidate list.")
        return []

    for rank, r in enumerate(sorted(combined, key=lambda r: r["p_crbn_binder"], reverse=True), 1):
        r["rank_04"] = rank
    for rank, r in enumerate(sorted(combined, key=lambda r: r["mgd_composite_score"], reverse=True), 1):
        r["rank_03"] = rank
    for r in combined:
        r["combined_rank"] = r["rank_03"] + r["rank_04"]

    combined.sort(key=lambda r: r["combined_rank"])
    n_active = max(1, round(len(combined) * active_fraction))
    active = combined[:n_active]

    with open(out_csv, "w", newline="") as f:
        fieldnames = ["name", "smiles", "p_crbn_binder", "p_crbn_glue", "p_ppil4_bind",
                      "mgd_composite_score", "rank_03", "rank_04", "combined_rank"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(active)

    print(f"\n=== Active candidates (top {active_fraction:.0%} by combined 03+04 rank) ===")
    print(f"{len(active)}/{len(combined)} candidates written to {out_csv}")
    return active


def main():
    print(f"Fetching all activity records for {TARGET_CHEMBL_ID} (human CRBN) ...")
    records = fetch_all_activities(TARGET_CHEMBL_ID)
    print(f"Raw activity records: {len(records)}")

    smiles_list, labels = dedupe_majority_vote(records)
    print(f"Unique compounds after SMILES dedup + majority vote: {len(smiles_list)}")
    print(f"Active (<= {POTENCY_THRESHOLD_NM/1000:.0f} uM): {sum(labels)}")
    print(f"Inactive: {len(labels) - sum(labels)}")

    X_rows, y_rows, kept_smiles = [], [], []
    for i, (smiles, label) in enumerate(zip(smiles_list, labels), 1):
        if i % 200 == 0:
            print(f"  ... fingerprinted {i}/{len(smiles_list)} compounds")
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            continue
        X_rows.append(fp)
        y_rows.append(label)
        kept_smiles.append(smiles)
    X = np.array(X_rows)
    Y = np.array(y_rows)
    print(f"\nFingerprinted dataset: X {X.shape}, Y {Y.shape} "
          f"(skipped {len(smiles_list) - len(kept_smiles)} unparseable SMILES)")

    # --- Random split ---
    from sklearn.model_selection import train_test_split
    rand_train_idx, rand_test_idx = train_test_split(
        np.arange(len(Y)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y,
    )
    random_results = train_and_evaluate(
        X[rand_train_idx], X[rand_test_idx], Y[rand_train_idx], Y[rand_test_idx],
        label="Random 80/20 split",
    )

    # --- Scaffold split ---
    scaffold_train_idx, scaffold_test_idx = scaffold_split(kept_smiles)
    scaffold_results = train_and_evaluate(
        X[scaffold_train_idx], X[scaffold_test_idx], Y[scaffold_train_idx], Y[scaffold_test_idx],
        label="Scaffold split",
    )

    print("\n=== Summary ===")
    print(f"{'Split':<15} {'AUC':>8} {'Precision':>10} {'Recall':>8}")
    print(f"{'Random':<15} {random_results['auc']:>8.4f} {random_results['precision']:>10.4f} "
          f"{random_results['recall']:>8.4f}")
    print(f"{'Scaffold':<15} {scaffold_results['auc']:>8.4f} {scaffold_results['precision']:>10.4f} "
          f"{scaffold_results['recall']:>8.4f}")
    print("\nThe scaffold-split numbers are the more trustworthy estimate of how this "
          "model generalizes to genuinely new chemotypes -- a gap between the two rows "
          "means the random split was leaking easy near-duplicate compounds into the test set.")

    # --- Score candidates.csv, if present, with a final classifier refit on
    # --- ALL labeled data (the random/scaffold splits above are only for
    # --- honest metric reporting, not for the model actually used to score
    # --- new candidates).
    if os.path.exists(CANDIDATES_CSV):
        print(f"\n=== Scoring {CANDIDATES_CSV} ===")
        final_clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
        )
        print(f"Training RandomForest ({N_ESTIMATORS} trees, final) ...")
        fit_start = time.time()
        final_clf.fit(X, Y)
        print(f"  ... done in {time.time() - fit_start:.1f}s")
        rows = score_candidates_file(final_clf, CANDIDATES_CSV, CANDIDATE_SCORES_CSV)
        n_active = sum(1 for r in rows if r["predicted_active"] == 1)
        print(f"Predicted active (P(binder) >= 0.5): {n_active}/{len(rows)}")

        if os.path.exists(MGD_SCORES_CSV):
            build_active_candidates(rows, MGD_SCORES_CSV, ACTIVE_CANDIDATES_CSV)
        else:
            print(f"\n{MGD_SCORES_CSV} not found -- run 03 first to also build "
                  f"{ACTIVE_CANDIDATES_CSV} for 06.")


if __name__ == "__main__":
    main()
