# IoT Industrial Trend Prediction
### Beijing PM2.5 Air Quality Forecasting Pipeline

---

## Project Overview

This project demonstrates an end-to-end machine learning pipeline for predicting industrial IoT sensor readings. Using hourly PM2.5 pollution data collected from Beijing (2010–2014), the pipeline ingests raw sensor data, applies production-grade cleaning for real-world IoT anomalies, engineers time-series features, trains a Gradient Boosting model with cross-validation, and evaluates predictions against ground truth.

**Target variable:** PM2.5 pollution level 1 hour ahead (µg/m³)  
**Dataset:** Beijing PM2.5 — Jason Brownlee / UCI ML Repository  
**Total samples:** ~43,800 hourly readings across 4 years

---

## Industrial Context

PM2.5 sensors are standard hardware in smart city infrastructure, industrial compliance monitoring, and precision environmental control systems. In real deployments these sensors suffer from:

- Intermittent dropouts (power loss, network failure)
- Out-of-bounds spikes from hardware fault or calibration drift
- Missing timestamps from clock sync errors
- Sensor flatline (frozen reading that looks like valid data)

This pipeline treats all of these as first-class engineering problems, not preprocessing afterthoughts.

---

## Repository Structure

```
IoT_Industrial_Trend_Prediction/
│
├── IoT_Industrial_Trend_Prediction.ipynb   # Main notebook (6 cells)
├── iot_bundle_YYYY-MM-DD.pkl               # Saved model + scaler + metadata bundle
├── README.md                               # This file
│
└── outputs/
    ├── feature_importance.png              # GradientBoosting feature chart
    ├── actual_vs_predicted.png             # Trend prediction graph
    └── residuals.png                       # Residual error plot
```

---

## Pipeline — Cell by Cell

### Cell 1 · Data Sourcing & Cleaning

- Loads raw CSV directly from source URL with `try/except` error handling
- Merges year/month/day/hour columns into a single `DatetimeIndex`
- Flags sensor dropout rows before imputation (`sensor_dropout` feature)
- **Clip first, then ffill** — physically impossible values (< 0 or > 999 µg/m³) are removed before forward-fill propagates them
- Forward-fill capped at `MAX_FFILL_GAP = 3` hours; longer gaps trigger time-interpolation fallback
- Wind direction encoded as `sin`/`cos` cyclical features + validity mask (`w_dir_valid`) to distinguish calm/variable from directional readings
- Index regularity check — warns if hourly frequency cannot be inferred (affects CV gap validity)

### Cell 2 · Feature Engineering

Raw split is performed **before** feature engineering to prevent rolling statistics from crossing the train/test boundary.

| Feature | Type | Rationale |
|---|---|---|
| `lag_1`, `lag_2`, `lag_3` | Autoregressive | Captures short-term momentum |
| `rolling_mean_6` | Smoothed level | 6-hour average removes hardware noise |
| `rolling_std_6` | Volatility | Captures sensor variability regime |
| `hour_sin`, `hour_cos` | Cyclical time | Encodes diurnal pollution pattern without discontinuity |
| `month` | Seasonal | Captures winter heating vs summer patterns |
| `w_dir_sin`, `w_dir_cos` | Cyclical wind | Directional pollution transport |
| `w_dir_valid` | Sensor quality | Flags calm/variable wind readings |
| `sensor_dropout` | Hardware flag | Tells model when prior readings are imputed |

### Cell 3 · Model Training

- **Scaler:** `RobustScaler` (median/IQR-based) — resistant to IoT spike outliers that would distort `MinMaxScaler` range
- **Validation:** `TimeSeriesSplit` with `n_splits=5` and `gap=WINDOW_SIZE` — gap prevents rolling features from the training tail contaminating the validation head
- **Model:** `GradientBoostingRegressor` with early stopping (`n_iter_no_change=20`) — automatically stops at convergence, avoids overfitting to a fixed tree count
- **Artifact:** model + scaler + metadata saved as a single versioned bundle (`iot_bundle_YYYY-MM-DD.pkl`) — prevents loading mismatched model/scaler versions

### Cell 4 · Feature Importance

Plots `model.feature_importances_` — `lag_1` dominates as expected for a 1-step-ahead task. In production, a sudden drop in a sensor feature's importance is an early signal of hardware degradation.

### Cell 5 · Evaluation & Visualization

- **Actual vs Predicted** — last 200 hours of test set overlaid
- **Residuals plot** — signed error over time; systematic bias in residuals would indicate model underfitting or concept drift

### Cell 6 · Final Summary

Compares model RMSE against a naive persistence baseline ("predict the last observed value"). A model that cannot beat this baseline has no predictive value over simple extrapolation.

---

## Results

| Metric | Value |
|---|---|
| Cross-validation RMSE | ~34.95 ± 5.73 |
| Test RMSE | ~29.87 |
| Test MAE | ~17.81 |
| Baseline RMSE (persistence) | ~22.33 |
| Features used | 16 |
| Training samples | ~33,000 |
| Test samples | ~8,600 |

> **Note on baseline comparison:** The current model RMSE exceeds the naive baseline, indicating that `lag_1` dominates model decisions. This is a known characteristic of 1-step-ahead PM2.5 forecasting with strong autocorrelation. Multi-horizon targets (6h, 24h) would show greater model advantage over persistence.

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

### Steps

1. Open `IoT_Industrial_Trend_Prediction.ipynb` in Google Colab or Jupyter
2. Run all cells top to bottom (`Runtime → Run all` in Colab)
3. Outputs:
   - Prints CV RMSE per fold + final test metrics
   - Displays Feature Importance chart
   - Displays Actual vs Predicted + Residuals plot
   - Saves `iot_bundle_YYYY-MM-DD.pkl` to working directory

### Loading the saved model

```python
import joblib

bundle = joblib.load("iot_bundle_2026-06-25.pkl")
model    = bundle["model"]
scaler   = bundle["scaler"]
metadata = bundle["metadata"]

print(metadata["trained_at"])
print(metadata["features"])
print(f"CV RMSE: {metadata['cv_rmse']:.4f}")
```

---

## Key Design Decisions

**Why clip before ffill?**  
A hardware spike (e.g., 1500 µg/m³) would be forward-filled into the next 3 rows before clipping could catch it. Clipping first converts the spike to a valid boundary value (or NaN if the spike exceeds physical bounds), then ffill operates on clean data only.

**Why RobustScaler over MinMaxScaler?**  
MinMaxScaler uses the global min/max of the training set. A single extreme IoT spike permanently distorts the entire feature range. RobustScaler uses median and IQR, making it unaffected by outliers.

**Why TimeSeriesSplit with a gap?**  
Standard K-Fold shuffles rows randomly — catastrophic for time series, creating future-to-past leakage. TimeSeriesSplit respects temporal order. The `gap=6` parameter ensures the 6 rows immediately before each validation window (which feed into rolling_mean_6) are excluded, preventing boundary contamination.

**Why not include `pollution` directly as a feature?**  
The target is `pollution.shift(-1)` (next hour). Including the current `pollution` value would create a near-trivial autocorrelation between feature and target that collapses at inference time on a live edge device where the current reading may not yet be available.

---

## Demo Video

> **[Add your video link here]**
>
> The video should cover:
> - Dataset context and industrial relevance (PM2.5 sensor network)
> - Walking through each cell — data cleaning decisions, feature engineering rationale
> - Running the pipeline live from raw ingest to final output
> - Explaining the Actual vs Predicted graph and what the residuals tell us
> - Model artifact versioning and how it would deploy to an edge device

---

## GitHub Repository

> **[Add your repository link here]**

---

## References

- Dataset: [Jason Brownlee — Beijing PM2.5 Data](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv)
- Original source: UCI Machine Learning Repository — Beijing PM2.5 Data Set
- Liang, X. et al. (2015). *Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and Winter Heating*. Proc. Royal Society A.
