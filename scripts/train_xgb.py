import pickle
import numpy as np

# Load the unpreprocessed data
with open('../data/unprocessed_data.pkl', 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)
    print('Data Loaded Successfully!')

# Apply Log for Price
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Load the model pipeline
with open('../models/best_xgb_model.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)
    print('Model Pipeline Loaded Successfully!')

# Fit the model pipeline
model_pipeline.fit(X_train, y_train_log)
print('Model Pipeline Fitted Successfully!')