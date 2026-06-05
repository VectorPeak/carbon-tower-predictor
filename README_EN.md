<div align="center">

# Carbon Tower Predictor

Industrial time-series modeling solution for carbonation tower outlet-liquid temperature prediction and predictive maintenance.

![python](https://img.shields.io/badge/python-3.10+-3776AB)
![pytorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C)
![seq2seq](https://img.shields.io/badge/Seq2Seq-direct--multi--step-blue)
![timeseries](https://img.shields.io/badge/time--series-forecasting-5C7CFA)
![maintenance](https://img.shields.io/badge/predictive-maintenance-green)

[简体中文](README.md) | English

</div>

```text
raw sensor data -> feature engineering -> diff dataset -> LSTM training -> rolling forecast
```

> Due to enterprise data confidentiality requirements, the raw datasets and intermediate data artifacts involved in this project are not publicly released. This repository focuses on the technical solution, modeling workflow, and engineering implementation.

---

## 0x01. Project Background

This project targets the carbonation tower process in the chemical industry. Its core objective is to forecast the outlet-liquid temperature for the next 30 minutes and use prediction residuals for anomaly warning and predictive maintenance.

The carbonation tower is a key unit in the ammonia-soda process. Outlet-liquid temperature reflects the thermal balance inside the tower. Abnormal temperature fluctuations may indicate degraded cooling efficiency, gas-feed imbalance, scaling, or other latent faults. Traditional DCS systems mainly provide real-time monitoring and fixed-threshold alarms, while this project focuses on trend-aware early warning.

The project adopts a Seq2Seq-style direct multi-step forecasting approach. It maps historical industrial sensor context to future state changes and learns equipment operating trends from multivariate time-series data.

---

## 0x02. Dataset Selection and Expansion

The original dataset contains one month of continuous operating data from multiple DCS sensor variables. The raw sampling frequency is approximately one point every two seconds, and the data is resampled to one-minute granularity for modeling.

The project evolved from a smaller set of basic variables to a broader process-variable set covering gas feed, reaction state, feed flow, cooling, discharge, and control-valve signals. After removing cumulative-flow variables that provide little dynamic information, the actual modeling input contains 13 effective features plus one target variable.

Typical data quality issues include quality-code contamination, frozen sensor values, timestamp misalignment, and timezone handling. Step 1 focuses on cleaning these issues and building a reliable wide table for downstream modeling.

---

## 0x03. Model and Technical Design

The project first evaluates several modeling routes, including machine-learning regression baselines, dedicated deep time-series architectures, anomaly-detection approaches, and direct multi-step LSTM forecasting.

The final baseline uses a Direct Multi-Step LSTM because it can output a full 30-minute prediction curve in a single forward pass, avoids recursive error accumulation, and is practical for industrial deployment and interpretation.

Feature engineering is a key part of the solution. Since many process variables are synchronous with the target after one-minute resampling, the pipeline introduces momentum features such as `diff1` and `diff3`, expanding the feature space from raw process values to change-rate signals.

The prediction target is also redesigned. Instead of directly forecasting absolute future temperature, the model predicts future temperature differences. The absolute temperature curve is reconstructed by:

```text
future_temperature = current_temperature + cumsum(predicted_difference)
```

This design reduces the burden of learning baseline drift and encourages the model to focus on dynamic process response.

---

## 0x04. Technical Architecture and Core Pipeline

```text
Raw CSV files
    -> Step 1: clean, resample, and merge into a wide table
    -> Step 2: lag-correlation analysis and feature-lag configuration
    -> Step 3: momentum features, normalization, and diff dataset construction
    -> Step 4: Direct Multi-Step LSTM training
    -> Step 5: diff restoration, rolling evaluation, and visualization
```

Main outputs:

```text
configs/feature_config.json
configs/norm_params_diff.json
data/merged/merged_wide_table.parquet
data/datasets/train_data_diff.npz
data/datasets/val_data_diff.npz
data/datasets/demo_data_diff.npz
artifacts/models/best_model_v4_diff.pth
artifacts/figures/
```

Data and model artifacts are ignored by Git where appropriate because they are either confidential or generated locally.

---

## 0x05. Value and Deployment Scenario

The project shifts the monitoring pattern from post-event alarm to pre-event trend awareness. In a practical DCS screen, operators can compare historical measured temperature, stitched rolling predictions, and the latest future forecast curve.

The rolling forecast design supports periodic refresh. Each inference predicts the next 30 minutes, while only the most reliable segment is stitched into the continuous prediction trajectory. This makes the output easier to interpret in an industrial operation interface.

---

## 0x06. Project Structure

```text
carbon-tower-predictor/
├── artifacts/
│   ├── figures/                  # generated evaluation figures
│   └── models/                   # local model weights, ignored by Git
├── configs/                      # feature and normalization configs
├── data/
│   ├── raw/                      # confidential raw CSV data, ignored by Git
│   ├── merged/                   # generated wide table, ignored by Git
│   └── datasets/                 # generated train/val/demo datasets, ignored by Git
├── docs/                         # step-by-step technical documents
├── scripts/                      # executable pipeline scripts
├── src/carbon_pipeline/          # CLI entrypoint
├── README.md
├── README_EN.md
└── pyproject.toml
```

---

## 0x07. Key Takeaways

1. Industrial data preprocessing determines the upper bound of model performance.
2. When no clear leading indicator exists, momentum features can provide useful dynamic signals.
3. Difference-based prediction is suitable for large-inertia industrial systems with baseline drift.
4. Loss design affects model behavior: robust losses help reduce the impact of spikes and outliers.
5. Visualization should serve operators, not only model developers.

---

## Running the Pipeline

Run the full pipeline from the project root:

```powershell
uv run carbon-pipeline -go
```

Or run steps individually:

```powershell
python scripts\step1_merge_16_vars.py
python scripts\step2_feature_selection_lag_correlation.py
python scripts\step3_dataset_diff_prediction.py
python scripts\step4_model_training.py
python scripts\step5_rolling_evaluation.py
```
