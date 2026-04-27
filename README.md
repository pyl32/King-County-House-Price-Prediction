# 🏠 King County House Price Prediction
## Spatial Analysis · Regression Modeling · Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest%20%7C%20MLP%20%7C%20MLR-orange)
![GeoPandas](https://img.shields.io/badge/GeoPandas-Spatial%20I/O-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Course](https://img.shields.io/badge/Course-SSCI%20575%20Spatial%20Data%20Science-purple)

---

## Overview

This project delivers a complete end-to-end house price prediction workflow for **King County, Washington** (21,613 sales, May 2014–May 2015) — from exploratory spatial analysis through three regression models.
---

## 📊 Model Results

| Model | R² | MAE ($) | RMSE ($) |
|-------|----|---------|---------|
| Multiple Linear Regression | 0.696 | $128,162 | $214,478 |
| Random Forest| 0.852 | $75,318 | $149,464 |
| **MLP Neural Network** | **0.882** | **$75,436** | **$133,495** |

> The MLP achieves **R²=0.882** 

---

## 🗺️ Study Area

**King County, Washington, USA** — encompassing Seattle and its suburbs (Bellevue, Kirkland, Redmond). Home to Amazon, Microsoft, and one of the USA's most dynamic real estate markets. Pronounced north/south price gradient driven by proximity to tech employment.

---

## 📂 Repository Structure

```
king-county-house-price-prediction/
├── README.md
├── King_County_House_Price_Prediction.ipynb   ← Main notebook
├── King_County_House_Price_Report.pdf         ← Full written report with figures
├── analysis.py                                ← Python script
├── data/
│   ├── king_county_house_sales.csv            ← Cleaned dataset (21,613 rows, 23 cols)
│   └── model_results.csv                      ← Model performance comparison
└── figures/
    ├── fig1_price_distribution.png            ← Map + histogram
    ├── fig2_correlation_heatmap.png           ← Full correlation matrix
    ├── fig3_price_correlations.png            ← Bar chart: correlations with price
    ├── fig4_scatter_predictors.png            ← Scatter: price vs top 4 features
    ├── fig5_boxplots.png                      ← Price by grade and view
    ├── fig6_spatial_density.png               ← Spatial price density map
    ├── fig7_actual_vs_predicted.png           ← All 3 models: actual vs predicted
    ├── fig8_model_comparison.png              ← R², MAE, RMSE bar charts
    ├── fig9_feature_importance.png            ← RF feature importance
    ├── fig10_residuals.png                    ← Residual analysis
    └── fig11_mlp_topology.png                 ← Network architecture + loss curve
```

---

## 🗃️ Dataset

**Source:** King County, WA house sales geodatabase (`Project2_dataset.gdb`)

| Attribute | Value |
|-----------|-------|
| Records | 21,613 |
| Period | May 2014 – May 2015 |
| Coordinate System | WGS 1984 Web Mercator |
| Null values | None |
| Price range | $75,000 – $7,700,000 |
| Median price | $450,000 |
| Mean price | $540,088 |

**Feature set (17 predictors):**

| Feature | Type | Description |
|---------|------|-------------|
| `bedrooms` | Integer | Number of bedrooms |
| `bathrooms` | Float | Number of bathrooms |
| `sqft_living` | Integer | Interior living area (sq ft) — r=0.70 with price |
| `sqft_lot` | Integer | Lot size (sq ft) |
| `floors` | Float | Number of floors |
| `waterfront` | Binary | Waterfront property |
| `view` | Ordinal | View quality (0=None, 4=Excellent) |
| `condition` | Ordinal | Property condition (1–5) |
| `grade` | Ordinal | Construction grade (1–13) — r=0.67 with price |
| `sqft_above` | Integer | Above-ground sq ft |
| `sqft_basement` | Integer | Basement sq ft |
| `house_age` | Integer | **Engineered:** 2015 − yr_built |
| `was_renovated` | Binary | **Engineered:** 1 if yr_renovated > 0 |
| `lat` | Float | Latitude — captures N/NE premium |
| `long` | Float | Longitude |
| `sqft_living15` | Integer | Avg living area of 15 nearest neighbors |
| `sqft_lot15` | Integer | Avg lot size of 15 nearest neighbors |

---

## ⚙️ Methods

### Exploratory Spatial Data Analysis (ESDA)
- Price distribution map (geographic scatter + histogram)
- Spatial price density (hot_r colormap revealing N/NE premium)
- Correlation matrix (Pearson r for all 17 features)
- Scatter plots: price vs top 4 predictors with trend lines
- Boxplots: price by construction grade and view rating

### Model 1: Multiple Linear Regression (MLR)
- Global linear fit, interpretable coefficients
- Weakness: constant coefficients ignore spatial nonstationarity
- Multicollinearity risk: `sqft_living` ↔ `sqft_above` (r=0.88)

### Model 2: Random Forest (Corrected)
```python
rf = RandomForestRegressor(
    n_estimators=200,   # was 28 -- far too low
    max_features=0.4,   # subsets ~7/17 features per split
    min_samples_leaf=2, # mild regularization
    random_state=42, n_jobs=-1
)
```

### Model 3: MLP Neural Network (Fixed)
```python
# STEP 1: Scale features (CRITICAL -- was missing in original)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# STEP 2: Train with corrected parameters
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),  # deeper than original (100,)
    activation='relu',
    learning_rate_init=0.001,
    max_iter=600,          # was 200 -- caused ConvergenceWarning
    early_stopping=True,   # prevents overfitting
    validation_fraction=0.1,
    random_state=42
)
mlp.fit(X_train_sc, y_train)  # use SCALED features
```

---

## 💡 Key Findings

**Geographic location dominates:** `lat` is the highest-importance feature in the Random Forest — confirming that King County's north/northeast price premium (Bellevue, Kirkland, Redmond tech corridor) drives prices more than most structural features.

**sqft_living and grade lead structural predictors:** Consistent with hedonic pricing theory and the correlation analysis (r=0.70 and r=0.67 respectively).

**Feature scaling is non-negotiable for MLP:** The original project's omission of StandardScaler produced R²≈0.42 — a gradient optimization failure, not a model limitation.

**Spatial nonstationarity is real:** GWR in ArcGIS Pro (original Project 2) achieved R²=0.814 by allowing coefficients to vary spatially, outperforming global MLR (R²=0.696) and confirming that one-size-fits-all regression misses local price drivers.

---

## 🚀 Getting Started

```bash
# Create environment
conda create -n king-county python=3.9
conda activate king-county
conda install -c conda-forge geopandas scikit-learn seaborn jupyterlab openpyxl

# Launch notebook
jupyter lab King_County_House_Price_Prediction.ipynb
```

**Note:** The notebook uses `data/Project2_dataset.gdb` — copy the original GDB from the project folder or adjust the path. The CSV export (`data/king_county_house_sales.csv`) can be used as a flat-file alternative.

---

## 📎 Report

See [`King_County_House_Price_Report.pdf`](King_County_House_Price_Report.pdf) for the full 20+ page written report including all figures, methodology details, results tables, and discussion.

---

## 📚 References

- ArcGIS Pro Documentation. (2023). *Geographically Weighted Regression (GWR)*. ESRI.
- Wang, Z., Wang, Y., Wu, S., & Du, Z. (2022). House Price Valuation Model Based on Geographically Neural Network Weighted Regression. *ISPRS International Journal of Geo-Information, 11*, 450.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR 12*, 2825–2830.

---
