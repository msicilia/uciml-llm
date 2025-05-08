## UCIML experiments with LLM


### Prepare and fetch datasets

A file `datasets.txt` with the listing of available UCIML datasets is required prior to download. This is currently extracted from the Notebook `explore_datasets.ipynb`.

Then, the datasets (only those that include `Classification` in the metadata) can be downloaded with the following

```
uv run get_datasets.py
```

This creates a subfolder of `datasets` for each dataset with the required files.

### Run experiments

```
uv run run_experiment.py
```

The results are saved in the `results` folder. 