import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets
import time
import os
import json

DATASET_FOLDER = './datasets'

datasets = pd.read_csv('datasets.txt', skiprows=5, names=['name', 'id'], header=None, 
                        delimiter=r'\s{2,}', skipinitialspace=True, engine='python')

for _, row in datasets.iterrows():
    dataset_name = row['name']
    try:
        dataset = fetch_ucirepo(dataset_name)
    except Exception as e:
        print(f"Error fetching dataset {dataset_name}: {e}")
        continue
    for k, v in dataset.items():
        print(f"{k}: \n{v}")
    folder = f"{DATASET_FOLDER}/{dataset_name}"
    if "Classification" in dataset["metadata"]["tasks"] and dataset["data"]["targets"] is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
            features = dataset["data"]["features"]
            features.to_csv(f"{folder}/features.csv", index=False)
            targets = dataset["data"]["targets"]
            targets.to_csv(f"{folder}/targets.csv", index=False)
            dataset["variables"].to_csv(f"{folder}/variables.csv", index=False)
            with open(f'{folder}/metadata.json', 'w') as f:
                json.dump(dataset["metadata"], f)
    time.sleep(10)

