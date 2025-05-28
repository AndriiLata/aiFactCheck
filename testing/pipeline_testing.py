import random
from pathlib import Path


def pick_test_instances(len_dataset, num_of_samples=5, used_indices_path="Datasets/used_indices.txt", ):
    used_indices = set()
    if Path(used_indices_path).exists():
        with open(used_indices_path, "r") as f:
            used_indices = set(map(int, f.read().splitlines()))

    all_indices = set(range(len_dataset))

    available_indices = list(all_indices - used_indices)

    if len(available_indices) < num_of_samples:
        print("All available claims have been tested.")
        exit()

    selected_indices = random.sample(available_indices, min(num_of_samples, len(available_indices)))

    with open(used_indices_path, "a") as f:
        for idx in selected_indices:
            f.write(f"{idx}\n")

    return selected_indices



import factkg_utils
import argparse
import requests
from tqdm import tqdm
import pandas as pd
import time

API_URL = "http://127.0.0.1:5000/verify"  # Adjust if your server is on a different host/port

def evaluate_via_api(samples: list[tuple[str, str]]) -> pd.DataFrame:
    results = []

    for claim, true_label in tqdm(samples, desc="Evaluating claims via API"):
        start_time = time.time()
        try:
            response = requests.post(API_URL, json={"claim": claim}, timeout=1000)
            if response.status_code == 200:
                raw_data = response.json()
                label_pred = raw_data.get("label", "NOT_ENOUGH_INFO")
                reason = raw_data.get("reason", "")
                triple = raw_data.get("triple", {})
                evidence = raw_data.get("evidence", [])
            else:
                label_pred = "NOT_ENOUGH_INFO"
                reason = ""
                triple = {}
                evidence = []
        except Exception as e:
            print(f"[!] Error for claim: {claim[:50]}... -> {e}")
            label_pred = "NOT_ENOUGH_INFO"
            reason = ""
            triple = {}
            evidence = []

        elapsed_time = time.time() - start_time  # end timer

        results.append({
            "claim": claim,
            "true_label": true_label,
            "predicted_label": label_pred,
            "triple": triple,
            "evidence": evidence,
            "reason": reason,
            "time_seconds": elapsed_time,
        })

    return pd.DataFrame(results)

from sklearn.metrics import classification_report, accuracy_score

def print_metrics(df: pd.DataFrame):
    """Displays classification metrics."""
    print("\nClassification Report:")
    print(classification_report(df["true_label"], df["predicted_label"],
                                labels=["Supported", "Refuted", "Not Enough Info"]))
    print(f"Accuracy: {accuracy_score(df['true_label'], df['predicted_label']):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to factkg_train.pickle")
    parser.add_argument("--samples", type=int, default=100, help="Number of random claims to test")
    parser.add_argument("--output", type=str, default="Datasets/factkg_api_results.pkl", help="Where to save results")
    parser.add_argument("--used_indices_path", type=str, default="Datasets/used_indices.txt", help="file for the used "
                                                                                                   "indices from "
                                                                                                   "dataset")

    args = parser.parse_args()

    print("[*] Loading dataset...")
    data = factkg_utils.load_factkg_dataset(args.file)
    len_dataset = len(data)

    print(f"[*] Sampling {args.samples} random claims...")
    test_indices = pick_test_instances(len_dataset, args.samples, args.used_indices_path)
    samples = factkg_utils.get_claims_by_indices(data, test_indices)

    print("[*] Sending samples to local /verify endpoint...\n")
    df_results = evaluate_via_api(samples)

    print_metrics(df_results)
    df_results.to_pickle(args.output)
    print(f"[*] Results saved to: {args.output}")
