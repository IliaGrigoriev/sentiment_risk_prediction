# Sentiment-Based Vehicle Safety Risk Analysis

This project combines NHTSA consumer complaints with recall campaigns to demonstrate how consumer sentiment and reported safety outcomes can support vehicle-risk exploration.

The workflow:

1. Loads and cleans NHTSA complaint and recall data.
2. Deduplicates overlapping recall exports.
3. Finds potential complaint-to-recall matches by make, model, and model year.
4. Extracts sentiment and safety-severity features from a configurable complaint sample.
5. Aggregates the features by vehicle and recall campaign.
6. Produces interactive Plotly figures for a risk-analysis demo.

## Project structure

```text
sentiment_risk_prediction/
├── data_managment/
│   ├── complaint_loader.py
│   ├── recall_loader.py
│   ├── data_cleaner.py
│   ├── data_matcher.py
│   ├── data_reporter.py
│   ├── data_store.py
│   └── data_manager.py
├── risk_analysis/
│   ├── risk_analyzer.py
│   └── risk_visualizer.py
├── const.py
├── features.py
├── sentiment_estimator.py
├── main.py
└── req.txt
```

`DataManager` coordinates loading, cleaning, matching, reporting, and persistence. `FeatureExtractor` creates complaint-level features. `RiskAnalyzer` performs feature-dependent aggregation, while `RiskVisualizer` generates the demo figures. The complete workflow is called only from `main.routine()`.

## Data

The expected source directory is:

```text
/home/ilia/Data/sent_risk_pred
```

Expected inputs:

```text
FLAT_CMPL.txt
RCL_FROM_2010_2014.csv
RCL_FROM_2015_2019.csv
RCL_FROM_2020_2024.csv
RCL_FROM_2025_2025.csv
RCL_FROM_2025_2026.csv
```

The complaints file is the tab-delimited NHTSA flat-file format. Recall files are loaded through the `RCL_FROM_*.csv` pattern, so additional exports matching that pattern are included automatically.

`Safercar_data.csv` is currently excluded from the workflow.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .srp
source .srp/bin/activate
pip install -r req.txt
```

The main dependencies are pandas, PyArrow, PyTorch, Transformers, and Plotly. GPU inference requires a CUDA-enabled PyTorch installation and a working NVIDIA driver.

## Running the analysis

Run:

```bash
.srp/bin/python main.py
```

The routine reports its four main stages:

```text
1/4 Cleaning complaints and recall documents
2/4 Matching complaints to potential recall campaigns
3/4 Extracting sentiment and safety features from a sample
4/4 Summarizing vehicle risk and creating demo figures
```

## Sampling

`FEATURE_SAMPLE_PERCENT` in `main.py` controls the percentage of complaint records that receive feature extraction:

```python
FEATURE_SAMPLE_PERCENT = 0.05
```

The value is a percentage, so `0.05` means **0.05%**, not 5%. With 2,255,013 complaints, this selects approximately 1,128 records.

Every sampled complaint receives both feature types. Sampling changes the number of analyzed complaints, not the number of features. Downstream risk summaries and figures are therefore sample-based.

## Features

### Negative sentiment

The sentiment model is:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

`sentiment_score` is the model's negative-class probability and ranges from 0 to 1. Higher values indicate more negative language.

The model uses CUDA device 0 when available and otherwise falls back to CPU. Eager attention is enabled to avoid a CUDA SDPA incompatibility observed with the current PyTorch and Transformers environment.

### Safety severity

`safety_severity_score` is an explainable ordinal measure based on reported outcomes:

| Score | Meaning |
|---:|---|
| 0 | No severe outcome recorded |
| 1 | Vehicle required towing |
| 2 | Crash reported |
| 3 | Fire reported |
| 4 | Injury or medical attention reported |
| 5 | Death reported |

When several outcomes apply, the highest severity is retained. This score describes recorded outcomes; it is not a trained prediction of future danger.

## Matching logic

Recall documents are deduplicated using:

```text
nhtsa_id + document_name + make + model + model_year
```

Distinct documents and vehicle variants within the same campaign are preserved. Complaints are then matched to campaign vehicles using standardized:

```text
make + model + model_year
```

The resulting fields include:

```text
has_matching_recall
matching_recall_count
matching_recall_ids
```

These are potential matches. They do not prove that a specific vehicle or VIN was included in a recall.

## Intermediate outputs

Generated files are stored under:

```text
/home/ilia/Data/sent_risk_pred/interm
```

Outputs include:

```text
cleaned_complaints.parquet
cleaned_recalls.parquet
matched_complaints.parquet
complaint_features.parquet
vehicle_risk_summary.parquet
recall_risk_summary.parquet
data_report.json
```

## Figures

Interactive Plotly figures are written to `figs/`:

### `sentiment_safety_risk_map.html`

Plots mean negative sentiment against mean safety severity for each make/model/model-year combination. Bubble size represents sampled complaint count, and color indicates whether a potential recall match exists.

### `potential_emerging_signals.html`

Ranks vehicles with severe sampled complaints but no potential recall match. These are investigation candidates, not evidence of an undiscovered defect.

### `recall_complaint_volume.html`

Shows sampled complaint volume potentially associated with recall campaigns. An identifier such as `18V268` means:

- `18`: campaign year 2018
- `V`: vehicle recall, not volts
- `268`: campaign sequence number

A complaint may be associated with several campaigns for the same vehicle, so campaign-level counts represent potential associations rather than confirmed causation.

## Limitations

- Feature extraction and all downstream figures use the configured sample.
- Make/model/year matching does not verify VIN or affected component.
- Negative sentiment and physical danger are separate concepts.
- Safety severity measures reported outcomes rather than predicted defect risk.
- The current recall CSV files do not contain exact recall dates, so before-versus-after recall analysis is not implemented.
- Results are suitable for a demonstration and exploratory analysis, not regulatory or safety decisions.
