import pandas as pd
import pickle
import pipeline_testing

# Small testing file to open previous results
with open("Datasets/factkg_api_results.pkl", "rb") as f:
    data = pickle.load(f)
    df = data["results"]
    metrics = data["metrics"]

print(df.iloc[0])
print(df.iloc[1])
#print(metrics)

pipeline_testing.print_metrics(df)
