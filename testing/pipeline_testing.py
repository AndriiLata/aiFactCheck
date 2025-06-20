import random
from pathlib import Path
from sklearn.metrics import classification_report
import utils
import argparse
import requests
from tqdm import tqdm
import pandas as pd
import pickle
from app.config import Settings

settings = Settings()

# Our server port adjust as necessary
API_URL = "http://127.0.0.1:5000/api/verify2"


# Picks random instances out of a test set, stores them in a file
# Input: len of dataset, num of samples to be drawn, a filepath to already used indices from the given dataset
# Return: List of indices
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


# Evaluates the given claims using http calls
# Input: A list of tuples with a claim and a label as strings
# Return: A pandas dataframe consisting of the tracked stats for the evaluation
def evaluate_via_api(samples) -> pd.DataFrame:
    results = []

    for sample in tqdm(samples, desc="Evaluating claims via API"):
        claim, true_label,*_=sample
        try:
            response = requests.post(API_URL, json={"claim": claim}, timeout=1000)
            if response.status_code == 200:
                raw_data = response.json()
                label_pred = raw_data.get("label", "NOT_ENOUGH_INFO")
                reason = raw_data.get("reason", "")
                triple = raw_data.get("triple", {})
                evidence = raw_data.get("evidence", [])
                timing_info = raw_data.get("timing_info") if settings.TIME_STEPS else None
            else:
                print(f"[!] Error for claim: {claim[:50]}... -> Response status code: {response.status_code}")
                label_pred = "Error"
                reason = ""
                triple = {}
                evidence = []
                timing_info = None
        except Exception as e:
            print(f"[!] Error for claim: {claim[:50]}... -> {e}")
            label_pred = "Error"
            reason = ""
            triple = {}
            evidence = []
            timing_info = None

        entry={
            "claim": claim,
            "true_label": true_label,
            "predicted_label": label_pred,
            "triple": triple,
            "evidence": evidence,
            "reason": reason,
        }

        if settings.TIME_STEPS and timing_info:
            entry["timing_info"] = timing_info

        results.append(entry)
    return pd.DataFrame(results)


# Function to display the metrics of a test run and calculate some stats
# confusion matrix
# Input: A pandas dataframe of the results
# Return: Classification report and confusion matrix
def print_metrics(df: pd.DataFrame):
    """Displays and returns classification metrics."""
    print("\nClassification Report:")
    report = classification_report(df["true_label"], df["predicted_label"],
                                   labels=["Supported", "Refuted", "Not Enough Info"],
                                   output_dict=False)
    print(report)

    # Confusion matrix
    print("\nPrediction Counts (Confusion Matrix):")
    confusion = pd.crosstab(df["true_label"], df["predicted_label"], rownames=['Actual'], colnames=['Predicted'],
                            dropna=False)
    print(confusion)

    if settings.TIME_STEPS and "timing_info" in df.columns:
        df["__effective_time__"] = df["timing_info"].apply(
            lambda x: x.get("0. total_time") if isinstance(x, dict) else None
        )
        avg_time = df["__effective_time__"].mean()
        print(f"\nAverage Time per Prediction: {avg_time:.3f} seconds")
        df.drop(columns="__effective_time__", inplace=True)


    return {
        "classification_report": report,
        "confusion_matrix": confusion.to_dict(),
        "average_time_per_prediction": avg_time
    }


# Main function to run the testing pipeline and display the results

# Important: To run the normal run.py script has to be started first and running for the same time
# To do that you need to enable allow multiple instances for both configurations

# Run with parameters --file Datasets/factkg_train.pickle --samples 100 --output --used_indices_path
# --file path to dataset file
# --samples number of samples to be tested; default=100
# --output path where output file should be stored; default="Datasets/factkg_api_results.pkl"
# --used_indices_path path to file which specifies which indices have already been used,
# After execution used indices stored here
# default="Datasets/used_indices.txt"

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

    #Switch to the relevant line of code
    data = utils.load_fever_dataset(args.file, drop_NEI=False)
    #data=utils.load_factkg_dataset(args.file)

    len_dataset = len(data)

    print(f"[*] Sampling {args.samples} random claims...")
    test_indices = pick_test_instances(len_dataset, args.samples, args.used_indices_path)
    samples = utils.get_claims_by_indices(data, test_indices)

    print("[*] Sending samples to local /verify endpoint...\n")
    df_results = evaluate_via_api(samples)


    metrics = print_metrics(df_results)
    with open(args.output, "wb") as f:
        pickle.dump({
            "results": df_results,
            "metrics": metrics
        }, f)

    print(f"[*] Results and metrics saved to: {args.output}")

