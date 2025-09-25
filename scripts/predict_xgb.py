import pickle
import numpy as np
import pandas as pd

# Load the unpreprocessed data
with open('../data/unprocessed_data.pkl', 'rb') as f:
    _, _, X_test, y_test = pickle.load(f)
    print('Data Loaded Successfully!')

# Random choice of a sample from the test set for prediction
idx = np.random.randint(0, X_test.shape[0])

# Sample For Prediction
X_sample = pd.DataFrame([X_test.iloc[idx]])

# Load the model pipeline
with open('../models/best_xgb_model.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# Fit the model pipeline
log_price = model_pipeline.predict(X_sample)
price = np.expm1(log_price)  # Apply expm1 to reverse the log1p transformation
print('Predicted Price: ', price[0])
print('Actual Price: ', y_test.iloc[idx])