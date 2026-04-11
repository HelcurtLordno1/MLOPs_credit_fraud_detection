# Fraud Detection Dataset Prep

This repository is now a simplified fraud-data preparation project.

The current workflow does one main job:

- read `fraudTrain.csv` and `fraudTest.csv`
- validate their columns and values
- write cleaned copies into `data/processed/`
- write a dataset summary into `reports/metrics/`

The older training, API, monitoring, and promotion modules were intentionally reduced and are no longer the active workflow.

## What To Do After Downloading This Folder

If you download or clone this project, do this first:

1. Open the project root folder.
2. Make sure these two dataset files exist in the root of the project:
   - `fraudTrain.csv`
   - `fraudTest.csv`
3. If they are missing, copy them into the project root manually.
4. Create a virtual environment and install dependencies.
5. Run the `prepare` command.

Expected root-level layout before running:

```text
fraud-detection/
|-- fraudTrain.csv
|-- fraudTest.csv
|-- README.md
|-- pyproject.toml
|-- dvc.yaml
|-- configs/
|-- data/
|-- src/
```

If `fraudTrain.csv` and `fraudTest.csv` are not present in the project root, the CLI will fail because it reads those exact files by default.

## Setup

### 1. Clone or download the project

```powershell
git clone <your-repo-url>
cd fraud-detection
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -e .[dev]
```

## How To Get The Data

This project now expects local CSV files, not the old zip-based PaySim input.

Put these files in the project root:

- `fraudTrain.csv`
- `fraudTest.csv`

That means the files should sit beside `README.md` and `pyproject.toml`, like this:

```text
fraud-detection/
|-- fraudTrain.csv
|-- fraudTest.csv
|-- README.md
|-- pyproject.toml
```

After that, run:

```powershell
$env:PYTHONPATH='src'
python -m fraud_detection.cli paths
python -m fraud_detection.cli prepare
```

What those commands do:

- `paths`: shows the resolved input and output locations
- `prepare`: validates the datasets and writes cleaned outputs

## Outputs

After `prepare` runs successfully, these files are created or refreshed:

- `data/processed/fraudTrain.csv`
- `data/processed/fraudTest.csv`
- `reports/metrics/dataset_summary.json`

Meaning:

- `data/processed/fraudTrain.csv`: cleaned validated copy of the training dataset
- `data/processed/fraudTest.csv`: cleaned validated copy of the test dataset
- `reports/metrics/dataset_summary.json`: row counts, fraud rate, amount stats, date range, and unique counts

## Current Folder Structure

```text
fraud-detection/
|-- fraudTrain.csv
|-- fraudTest.csv
|-- README.md
|-- pyproject.toml
|-- dvc.yaml
|-- configs/
|   |-- data.yaml
|   |-- monitoring.yaml
|   |-- serve.yaml
|   |-- train.yaml
|-- data/
|   |-- interim/
|   |-- monitoring/
|   |-- processed/
|   |   |-- fraudTrain.csv
|   |   |-- fraudTest.csv
|   |-- raw/
|-- deployment/
|-- models/
|   |-- registry/
|   |-- trained/
|-- reports/
|   |-- drift/
|   |-- figures/
|   |-- metrics/
|   |   |-- dataset_summary.json
|-- scripts/
|-- src/
|   |-- fraud_detection/
|   |   |-- cli.py
|   |   |-- config.py
|   |   |-- data/
|   |   |   |-- schema.py
|   |   |-- utils/
|   |   |   |-- paths.py
|   |   |   |-- mlflow_utils.py
|-- streamlit_app/
|-- tests/
```

## What Each Folder Means

- `configs/`: configuration files. Only `configs/data.yaml` matters for the current workflow.
- `data/`: data storage area.
- `data/raw/`: old raw-data location from the previous version of the project.
- `data/interim/`: placeholder folder for temporary data if you add preprocessing later.
- `data/monitoring/`: placeholder folder for monitoring-related data if the project grows again.
- `data/processed/`: current validated output datasets.
- `deployment/`: old deployment-related files from the previous full MLOps version.
- `models/`: old model artifact folders kept in the repo structure.
- `models/trained/`: where trained model files would go in a larger version of the project.
- `models/registry/`: where model registration metadata would go in a larger version of the project.
- `reports/`: generated reports.
- `reports/metrics/`: current summary output location.
- `reports/figures/`: placeholder for plots if you add them later.
- `reports/drift/`: placeholder for drift reports from the old workflow.
- `scripts/`: utility scripts that are not part of the main CLI flow.
- `src/`: application source code.
- `src/fraud_detection/`: main Python package.
- `src/fraud_detection/data/`: dataset schema definitions.
- `src/fraud_detection/utils/`: reusable helper functions.
- `streamlit_app/`: old UI folder from the previous version.
- `tests/`: existing test files. Some of them still target the older workflow and may need updates.

## What To Put In Each Important File

### Root dataset files

- `fraudTrain.csv`: full training dataset
- `fraudTest.csv`: full test dataset

These should contain the original transaction rows.

Expected columns:

- `Unnamed: 0`
- `trans_date_trans_time`
- `cc_num`
- `merchant`
- `category`
- `amt`
- `first`
- `last`
- `gender`
- `street`
- `city`
- `state`
- `zip`
- `lat`
- `long`
- `city_pop`
- `job`
- `dob`
- `trans_num`
- `unix_time`
- `merch_lat`
- `merch_long`
- `is_fraud`

### `configs/data.yaml`

Use this file to define where the input files are and where outputs should be written.

Current meaning of fields:

- `train_csv`: location of `fraudTrain.csv`
- `test_csv`: location of `fraudTest.csv`
- `processed_dir`: folder for cleaned output files
- `reports_dir`: folder for summary files
- `target_column`: fraud label column name
- `index_column`: renamed index column in processed output
- `summary_name`: output summary file name
- `sample_rows`: optional limit for a smaller run

### `src/fraud_detection/cli.py`

This is the active entry point.

It should contain:

- argument parsing
- config loading
- CSV reading
- validation calls
- processed output writing
- summary writing

Current commands:

- `paths`
- `prepare`

### `src/fraud_detection/data/schema.py`

This file should contain:

- expected dataset column names
- dtype definitions for reading CSV
- validation rules for values and formats

Examples of checks currently enforced:

- `is_fraud` must be `0` or `1`
- `gender` must be `F` or `M`
- `state` must look like a 2-letter uppercase code
- latitude and longitude must be in valid ranges
- dates must match expected formats

### `src/fraud_detection/utils/paths.py`

This file should contain path helpers only.

Current responsibility:

- locate the project root
- create directories if they do not exist

### `src/fraud_detection/utils/mlflow_utils.py`

This is a helper file kept from the older project structure.

Right now it is not part of the active simplified flow. Keep utility helpers here only if you expand the project later.

## Example Run

Small smoke run:

```powershell
$env:PYTHONPATH='src'
python -m fraud_detection.cli prepare --sample-rows 2000
```

Full run:

```powershell
$env:PYTHONPATH='src'
python -m fraud_detection.cli prepare
```

## Notes

- `dvc.yaml` still reflects the older full pipeline structure and is not the active workflow now.
- Many folders remain in the repository to preserve structure, but only the dataset preparation flow is active.
- If you want, the next cleanup step should be updating `dvc.yaml` and removing old tests so the repo matches the simplified README completely.
