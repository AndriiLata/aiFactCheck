import pandas as pd
import pickle
import pipeline_testing

# Small testing file to open previous results
with open("Datasets/factkg_api_results.pkl", "rb") as f:
    data = pickle.load(f)
    df = data["results"]
    metrics = data["metrics"]

with open("Datasets/factkg_api_results400.pkl", "rb") as f:
    data400 = pickle.load(f)
    df400 = data400["results"]
    metrics400 = data400["metrics"]

print(df.iloc[0, 0])
print(df.iloc[1])
#print(metrics)

combined = pd.concat([df, df400], axis=0)

pipeline_testing.print_metrics(combined)
