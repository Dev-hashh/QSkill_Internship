# House Price Prediction

A machine learning notebook that predicts median house values using the California Housing dataset, comparing Linear Regression and Random Forest models.

## Overview

This project walks through a complete regression workflow: exploratory data analysis, outlier detection, feature scaling/transformation, model training, and evaluation — finishing with a comparison between a baseline Linear Regression model and a Random Forest Regressor.

## Dataset

- **Source:** `sklearn.datasets.fetch_california_housing` (built into scikit-learn, no external download needed)
- **Target variable:** `MedHouseVal` — median house value for California districts
- **Features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

## What the Notebook Does

1. **Data Loading & Inspection** — loads the dataset into a pandas DataFrame and prints head/describe/info summaries.
2. **Exploratory Data Analysis**
   - Correlation heatmap of all features
   - Geographic scatter plot of house values by latitude/longitude
3. **Outlier Detection** — boxplot and Z-score analysis on `MedInc` (median income) to flag outliers.
4. **Feature Engineering**
   - `RobustScaler` applied to `MedInc` (scaled version stored as `MedInc_scaled`)
   - Log transformation (`np.log1p`) applied to `MedInc` (stored as `MedInc_log`) to reduce skew, with before/after boxplot comparison
5. **Train/Test Split** — 80/20 split (`random_state=42`)
6. **Linear Regression**
   - Trains a baseline model
   - Reports RMSE, MAE, and R² Score
   - Plots predicted vs. actual values
   - Residual analysis: residuals vs. predicted, residual distribution, and Q-Q plot
7. **Random Forest Regressor**
   - 200 trees, full depth, using all CPU cores (`n_jobs=-1`)
   - Reports RMSE, MAE, and R² Score
   - Predicted vs. actual plot
8. **Model Comparison** — side-by-side RMSE/MAE/R² comparison between Linear Regression and Random Forest
9. **Feature Importance** — bar chart of the top 10 most important features from the Random Forest model

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

## Usage

1. Open `House_Price_Prediction.ipynb` in Jupyter Notebook / JupyterLab / VS Code.
2. Run all cells in order (no external data files required — the dataset is fetched automatically via scikit-learn).

## Results

The notebook prints and visualizes RMSE, MAE, and R² for both models, letting you directly compare how well a simple linear model performs against an ensemble Random Forest approach on this dataset, along with which features drive predictions most.

## Notes

- The first cell (`print("hello")`) is a leftover sanity check and can be safely removed or ignored.
- Random seed (`random_state=42`) is fixed throughout for reproducibility.


## 👤 Author

**QSkill Internship Project**  
Part of the Machine Learning Internship Task Series.