# Car Price Prediction

A machine learning project to predict used car prices using classical and ensemble methods. This repository includes data exploration, cleaning, preprocessing, feature engineering, model training, and evaluation, with a focus on XGBoost as the primary model.

## Project Structure

```
car-price-prediction-main/
├── README.md                    # Project overview and instructions
├── requirements.txt             # Python dependencies
├── data/
│   ├── train-data.csv           # Raw dataset
│   ├── semi_cleaned_data.csv    # Cleaned dataset
│   ├── unprocessed_data.pkl     # Pickled train/test splits
│   ├── preprocessed_data.pkl    # Cached preprocessed data
├── models/
│   └── best_xgb_model.pkl       # Trained XGBoost pipeline
├── notebooks/
│   ├── 01- Data Exploration.ipynb                # Dataset inspection and visualization
│   ├── 02- Data Cleaning & Feature Engineering.ipynb  # Data cleaning and feature creation
│   ├── 03- Data Preprocessing and Base Model.ipynb   # Preprocessing pipeline and baseline models
│   └── 04- Modelling.ipynb                      # Advanced model training and evaluation
├── reports/
│   └── first-exploration.html    # Automated data profiling report
└── scripts/
    ├── train_xgb.py             # Train XGBoost pipeline
    └── predict_xgb.py           # Sample prediction with saved model
```

## Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/a7madmostafa/car-price-prediction.git
cd car-price-prediction-main
```

### 2. Set Up the Environment

**Option A: Using `uv` (Recommended)**
```bash
pip install uv
uv venv car-price
source car-price/bin/activate  # Windows: car-price\Scripts\activate
uv pip install -r requirements.txt --link-mode=copy
```

**Option B: Using `pip`**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Notebooks
- Open the `notebooks/` folder in Jupyter Notebook or VS Code.
- Select the Python kernel (`car-price` for `uv` or `.venv` for `pip`).
- Follow the workflow in the notebooks (01 to 04).

### 4. Train the Model
```bash
cd scripts
python train_xgb.py
```

### 5. Make a Sample Prediction
```bash
cd scripts
python predict_xgb.py
```

## Workflow Overview

1. **Data Exploration** (`01- Data Exploration.ipynb`)
   - Inspect the raw dataset (`train-data.csv`).
   - Analyze target distribution and feature correlations.
   - Generate initial visualizations (e.g., histograms, correlation heatmaps).

2. **Data Cleaning & Feature Engineering** (`02- Data Cleaning & Feature Engineering.ipynb`)
   - Handle missing values and inconsistent units.
   - Create features like car age.
   - Save cleaned dataset (`semi_cleaned_data.csv`).

3. **Data Preprocessing and Base Model** (`03- Data Preprocessing and Base Model.ipynb`)
   - Build a preprocessing pipeline (imputation, encoding, scaling).
   - Train and evaluate baseline models (e.g., linear regression).
   - Save preprocessed data (`preprocessed_data.pkl`).

4. **Modelling** (`04- Modelling.ipynb`)
   - Train advanced models (Random Forest, XGBoost, Ridge).
   - Evaluate using metrics like R² and RMSE.
   - Save the best pipeline (`best_xgb_model.pkl`).

## Requirements

Key dependencies (see `requirements.txt`):
- `numpy`, `pandas`
- `matplotlib`, `seaborn`, `plotly`
- `scikit-learn`, `category-encoders`
- `xgboost`, `optuna`
- `missingno`, `ydata-profiling`, `ipywidgets`

Install with:
```bash
pip install -r requirements.txt
```

## Scripts

- **train_xgb.py**: Loads data, applies log-transform to the target, fits an XGBoost pipeline, and saves it.
- **predict_xgb.py**: Loads the saved pipeline, selects a test sample, and compares predicted vs. actual prices.

## Reports

- **first-exploration.html**: Automated report (`ydata-profiling`) with summary statistics, correlations, and missing value visualizations.

## License

No license is currently specified. For open-source usage, consider adding an [MIT License](https://opensource.org/licenses/MIT).

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.
