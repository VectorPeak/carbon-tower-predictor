<div align="center">

# Carbon Tower Predictor | Predictive Maintenance System for Carbonation Tower Outlet-Liquid Temperature

Time-series modeling solution for industrial carbonation tower outlet-liquid temperature prediction and predictive maintenance

![python](https://img.shields.io/badge/python-3.10+-3776AB)
![pytorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C)
![seq2seq](https://img.shields.io/badge/Seq2Seq-direct--multi--step-blue)
![timeseries](https://img.shields.io/badge/time--series-forecasting-5C7CFA)
![maintenance](https://img.shields.io/badge/predictive-maintenance-green)

[Simplified Chinese](README.md) | English

</div>

```text
raw sensor data -> feature engineering -> diff dataset -> LSTM training -> rolling forecast
```

> Due to enterprise data confidentiality requirements, the raw datasets and intermediate data artifacts involved in this project are not publicly released. This document only discusses the technical solution, modeling workflow, and engineering implementation.

## 0x01. Project Background

This project targets the carbonation tower process in the chemical industry. Its core objective is to perform rolling prediction of **carbonation outlet-liquid temperature** over the next 30 minutes, and to support anomaly warning and predictive maintenance through prediction residuals.

The carbonation tower is a core unit in the ammonia-soda process for soda ash production. Its outlet-liquid temperature directly reflects the thermal balance state inside the tower. Abnormal temperature fluctuations often indicate potential faults such as declining cooling-system efficiency, gas-feed ratio imbalance, or tower scaling. Traditional DCS systems mainly provide real-time monitoring and fixed-threshold alarms, but cannot provide **trend anticipation**.

The project adopts a Seq2Seq-style design: it maps historical industrial sensor context to future state changes, and learns equipment operating trends through direct multi-step forecasting, enabling earlier perception of equipment degradation trends.

---

## 0x02. Dataset Selection and Expansion

### 2.1 Data Scale and Source

- **Time span**: one month of continuous operating data, approximately 30 days
- **Original sampling frequency**: one point every 2 seconds; each variable has roughly one million rows; the total dataset contains about 13 million records
- **After downsampling**: one point per minute; each variable contains roughly 43,200 rows
- **Data format**: CSV exported from the industrial DCS system, separated by `|` or `,`, including timestamp, quality flag, data value, and related fields

### 2.2 Dataset Evolution: From 5 Variables to 16 Variables

The project went through two important dataset expansion stages:

**Stage 1 (5 variables)**:
The initial version used only five basic measurement points: outlet tail-gas pressure, CO2 inlet pressure, carbonation outlet-liquid temperature (target), middle reaction-section temperature, and liquid level.

**Stage 2 (16 variables)**:
The five initial variables were all found to be **synchronously correlated (Lag=0)** with the target temperature, lacking physically meaningful leading indicators. This caused prediction lag in the model. The dataset was therefore expanded to 16 measurement points covering the complete carbonation tower process loop:

| Category | Variable | Physical Meaning |
|------|--------|---------|
| **Gas feed system** | CO2 inlet pressure | Reaction driving force |
| **Reaction system** | Middle reaction-section temperature | Reaction progress indicator |
| **Feed system** | Ammonia mother liquor II feed flow / valve position | Feed control |
| **Feed system** | Carbonated ammonia mother liquor II feed flow / valve position | Feed control |
| **Cooling system** | Cooling-water outlet temperature from water tank | Heat-exchange efficiency |
| **Cooling system** | Cooling-water pressure | Cooling-system state |
| **Discharge system** | Outlet-liquid flow | Discharge load |
| **Target variable** | **Carbonation outlet-liquid temperature** | **Prediction target** |
| **Control variable** | Outlet-liquid temperature control-valve position | Temperature-control actuator |
| **Auxiliary variables** | Liquid level, outlet tail-gas pressure | Tower internal state |

> **Key decision**: three cumulative-flow variables were removed because they increase monotonically and carry little dynamic information. The actual model input contains **13 effective features + 1 target variable**.

![image.png](https://img.vectorpeak.cn/obsidian/2026/05-06/20260605162656135.png?imageSlim)

### 2.3 Data Quality Work

Industrial field data contains typical "dirty data" traps. This project addresses them one by one:

| Issue Type | Symptom | Solution |
|---------|------|---------|
| **Quality flag misread** | DCS quality flag `192` is misread as a temperature value, causing the column mean to become 192°C | Identify the `datavalue` column and exclude the `dataquality` column |
| **Dead value / frozen value** | When sensor communication is interrupted, the system keeps the last valid value unchanged | Detect N consecutive identical sampled values and force them to `NaN` |
| **Timestamp alignment** | Sampling times differ by seconds across sensors | Outer join + forward fill + one-minute resampling |
| **Timezone issue** | Timestamp contains `+08:00` timezone information | Strip timezone information and convert to local-time index |

---

## 0x03. Model and Technical Selection Process

### 3.1 Stage 1: Migration and Reflection on Sequence Modeling

**Initial idea**: use the historical sensor window as model input and attempt to forecast the future temperature curve through sequence modeling.

**Observed bottlenecks**:

- Industrial data consists of **continuous floating-point values**, and autoregressive prediction easily accumulates errors
- One month of data is relatively thin for large-scale deep models and can lead to overfitting
- Directly predicting future values step by step can cause **multi-step prediction errors to amplify**

**Conclusion**: sequence modeling is useful, but it must be deeply adapted to the **continuity, physical inertia, and noise characteristics** of industrial time series.

### 3.2 Stage 2: Model Selection Debate Across Four Routes

Before finalizing the architecture, four technical routes were evaluated:

| Route | Representative Models | Advantages | Limitations | Suitability for This Project |
|------|---------|------|------|------------|
| **Machine-learning regression** | XGBoost / LightGBM | Strong on small data, very fast training, good interpretability | Cannot output continuous forecast curves and loses temporal micro-shape | Can serve as a baseline, but not the main model |
| **Dedicated deep time-series models** | TCN / PatchTST | Fast parallel computation, strong long-range dependency modeling | Requires more data and complex tuning | Can be upgraded later when more data is available |
| **Anomaly-detection models** | Autoencoder | Does not require forecasting the future; detects anomalies directly | Cannot provide forward-looking trend information | Auxiliary only |
| **Direct multi-output LSTM** | **Direct Multi-Step LSTM** | Outputs a 30-minute curve in one pass, avoids autoregressive error accumulation, proven industrial practicality | Has lag in response to sudden changes | **Final choice** |

**Final decision**: use the **Direct Multi-Step LSTM architecture** as the baseline model, for three reasons:

1. Thirty days of minute-level data, with roughly 4,000+ sliding-window samples (`43200 x 14`), is sufficient to support LSTM training.
2. LSTM has relatively low deployment and explanation cost in industrial DCS environments.
3. Later **feature engineering** and **loss-function redesign** can compensate for LSTM's slower response to abrupt changes.

### 3.3 Stage 3: Feature Engineering: From Absolute Values to Momentum

**Core finding**: Step 2 lag-correlation analysis shows that all variables reach maximum correlation with the target temperature at **Lag=0 (synchronous)**, with no obvious physically leading indicators.

**Root cause**: the carbonation tower is a large-inertia system. Physical delays occur at the second level and are smoothed out after one-minute downsampling. Stable-state operation also dominates the data.

**Solution: momentum feature injection**:

- Construct **one-minute change rate (`diff1`)** and **three-minute change rate (`diff3`)** for each raw feature
- Expand feature dimensionality from 13 to **39 dimensions (13 raw + 13 diff1 + 13 diff3)**
- **Physical meaning**: even if absolute values are synchronous, their change rates may still indicate upcoming trend turns, such as valve-position adjustment -> flow change -> temperature response

### 3.4 Stage 4: Prediction Target Redesign: Difference Prediction

**Key innovation**: instead of predicting the **absolute temperature** for the next 30 minutes, the model predicts the **temperature change amount (difference)**.

```text
Traditional design: input [past 30 minutes] -> predict [absolute temperature over next 30 minutes]
This design:        input [past 30 minutes] -> predict [minute-by-minute temperature change over next 30 minutes]
```

**Advantages**:

- Reduces the burden of baseline drift. Different batches and seasons may have different temperature baselines, but physical response patterns remain similar.
- Allows the model to focus on learning "disturbance response" rather than memorizing the current temperature.
- Restoration formula: $T_{future} = T_{current} + cumsum(\Delta \hat{y})$

### 3.5 Stage 5: Loss Function Evolution: From MSE to Huber + Difference Loss

**First training attempt (MSE / Huber Loss)**:

- Prediction was excellent in stable-state segments, but showed severe lag in **transient segments (sharp temperature drops)**.
- The model tended to predict a "safe average value", which is the regression-to-the-mean effect.

**Second training attempt (introducing TrendAwareLoss)**:

$$
L_{total} = \alpha L_{Huber}(y, \hat{y}) + \beta L_{Huber}(\Delta y, \Delta \hat{y})
$$

- Penalizes not only inaccurate temperature values, but also inaccurate temperature-change trends or slopes.
- Forces the model to follow steep waveform changes during sharp temperature drops.

### 3.6 Stage 6: Evaluation Visualization: Trajectory Stitching

The original evaluation approach, which drew one independent prediction line every five minutes, made it difficult for operators to compare prediction and ground truth intuitively.

![image.png](https://img.vectorpeak.cn/obsidian/2026/05-06/20260605162726508.png?imageSlim)

**Innovation: trajectory stitching**:

- Each inference outputs a 30-minute forecast, but only the first `ROLL_STRIDE` minutes, such as 10 minutes, are treated as the reliable segment. In practical use, the system predicts 30 minutes ahead each time, then refreshes and updates the forecast curve every 10 minutes.
- Multiple reliable segments are stitched end to end along the time axis to form a **continuous prediction curve fully aligned with the true values**.
- The remaining unverified segment is shown as a dashed "future outlook" prediction.

**Run results**

***Final effect***

![image.png](https://img.vectorpeak.cn/obsidian/2026/05-06/20260605162739903.png?imageSlim)

![image.png](https://img.vectorpeak.cn/obsidian/2026/05-06/20260605162749827.png?imageSlim)

***Local MAE evaluation***

![image.png](https://img.vectorpeak.cn/obsidian/2026/05-06/20260605162801334.png?imageSlim)

Online rolling prediction mode, updated every 10 minutes:
  MAE (Mean Absolute Error): 0.2592 ℃

---

## 0x04. Technical Architecture and Core Workflow

```text
Raw CSV files (16 variables, 2-second sampling, about 1 million rows per variable)
    |
    v
Step 1: Data cleaning and wide-table merge
    |-- Identify the datavalue column to avoid the 192 quality-flag trap
    |-- Detect dead values and replace with NaN when values remain unchanged for 5 consecutive points
    |-- Align timestamps and downsample to 1-minute resolution through mean aggregation
    |-- Output: merged_wide_table.parquet (43,200 rows x 14 columns)
    |
    v
Step 2: Leading-indicator discovery and feature selection
    |-- Compute -30 to +30 minute lag correlations between 13 features and the target
    |-- Remove lagging indicators (Lag < 0) and weak correlations (|corr| < 0.35)
    |-- Find that features are mostly synchronous or weakly correlated, then start momentum feature engineering
    |-- Output: feature_config.json (feature -> lag mapping)
    |
    v
Step 3: Momentum enhancement and dataset construction
    |-- Add diff1 / diff3 momentum features (13 dimensions -> 39 dimensions)
    |-- Shift feature columns according to the configuration to prevent data leakage
    |-- Time-series split: train (14 days) | validation (2 days) | demo (7 days)
    |-- Z-score normalization using training-set statistics only
    |-- Sliding-window slicing: input [30, 39] -> output [30], stride 5 minutes
    |
    v
Step 4: Direct multi-output LSTM training
    |-- Model: LSTM(39 -> 64 -> 30) + Linear(64 -> 30)
    |-- Loss: TrendAwareLoss (dual penalty on absolute value and difference trend)
    |-- Optimization: Adam + ReduceLROnPlateau + EarlyStopping
    |-- Output: best_lstm_model.pth
    |
    v
Step 5: Difference restoration and rolling evaluation
    |-- Load normalization parameters -> inverse transform -> restore absolute temperature with cumsum
    |-- Trajectory stitching: take reliable segments every 10 minutes and stitch a continuous curve
    |-- Metrics: MAE and RMSE in real physical units (℃)
    |-- Visualization: historical truth + stitched prediction trajectory + dashed future outlook
```

---

## 0x05. Final Value and Deployment Scenarios

### 5.1 Prediction Performance

| Scenario | Prediction Horizon | Error Metric | Value Interpretation |
|------|-------------|---------|---------|
| **Stable operation** | Next 30 minutes | MAE ≈ 0.16°C | The predicted curve almost overlaps the true curve and can be used for **trend confirmation** |
| **Transient fluctuation** | Next 30 minutes | MAE ≈ 2.06°C | Captures trend direction with slight lag and can support **degradation warning** |
| **Long-range divergence** | 15-30 minutes | MAE gradually increases | Longer forecasts become more conservative, consistent with physical uncertainty |

### 5.2 Business Value

1. **From post-event alarms to pre-event warning**
   - Traditional DCS: alarms only after temperature exceeds a threshold, when loss has already occurred.
   - This system: predicts temperature trends 30 minutes ahead and triggers warnings during early deviation.

2. **Dynamic residual-based anomaly detection**
   - During normal operation, prediction residuals fluctuate randomly around zero.
   - During equipment degradation, residuals show **systematic one-directional deviation**, such as predicted values being consistently higher than true values for five consecutive minutes.
   - Dynamic thresholds based on residual distribution, such as `μ ± 3σ`, help reduce false alarms from fixed thresholds.

3. **DCS screen integration**
   - Blue solid line: historical measured temperature
   - Red solid line: verified stitched prediction trajectory, updated every 10 minutes
   - Red dashed line: latest 30-minute future outlook
   - Operators can directly see "how the system thinks the temperature will change"

4. **Process optimization assistance**
   - Analyze the lag relationship between valve opening and temperature response to optimize PID parameters.
   - Identify the response delay between cooling-water pressure fluctuations and temperature, supporting maintenance-cycle planning.

---

## 0x06. Project File Structure

**Due to enterprise data confidentiality requirements, the raw datasets and intermediate data artifacts involved in this project are not publicly released. This document only discusses the technical solution, modeling workflow, and engineering implementation.**

```text
industrial_predictor/
├── raw_data/                          # Raw CSV files for 16 variables
│   ├── A-1CO2 gas inlet pressure.csv
│   ├── A-1 outlet tail-gas pressure.csv
│   ├── A-1 middle reaction-section temperature.csv
│   ├── ...
│   └── A-1 carbonation outlet-liquid temperature.csv     # Target variable
│
├── processed_data/                    # Step 1 intermediate output after single-variable downsampling
│   ├── CO2 inlet pressure.parquet
│   ├── outlet tail-gas pressure.parquet
│   └── ...
│
├── merged_data/                       # Step 2 output
│   └── merged_wide_table.parquet      # 14-column wide table, 13 features + 1 target
│
├── config/                            # Configurations and parameters
│   ├── feature_config.json            # Step 2: feature lag configuration
│   ├── norm_params.json               # Step 3: Z-score normalization parameters
│   └── lag_correlation_plot.png       # Step 2: lag-correlation visualization
│
├── dataset/                           # Step 3 output, directly readable by the model
│   ├── train_data.npz                 # Training set (X, Y)
│   ├── val_data.npz                   # Validation set (X, Y)
│   └── demo_data.npz                  # Demo set (X, Y)
│
├── models/                            # Step 4 and Step 6 output
│   ├── best_lstm_model.pth            # Best model weights
│   ├── loss_curve.png                 # Training loss curve
│   └── prediction_vs_real.png         # Prediction vs. ground truth figure
│
├── step1_merge_16_vars.py             # Data cleaning and wide-table merge
├── step2_feature_selection_lag_correlation.py  # Lag-correlation analysis and feature selection
├── step3_dataset_diff_prediction.py   # Sliding-window slicing and normalization
├── step4_model_training.py # Direct multi-output LSTM training
├── step5_rolling_evaluation.py # Difference restoration, rolling evaluation, and visualization
│
└── README.md                          # This file
```

---

## 0x07. Key Lessons Learned

1. **Industrial data preprocessing accounts for most of the work**: quality-flag identification, frozen-value cleaning, and timestamp alignment all determine the upper bound of model performance.
2. **When no leading indicator exists, create momentum indicators**: when physical delay is smoothed out by sampling frequency, change rate (`diff`) becomes the last available forward-looking signal.
3. **Difference prediction is well suited to large-inertia industrial systems**: predicting "change" is more physically intuitive than predicting "absolute value".
4. **The loss function shapes model behavior**: MSE makes the model conservative, while Huber + difference loss makes it more responsive.
5. **Visualization must serve operators**: a continuous stitched trajectory is more valuable in engineering practice than scattered prediction bands.
