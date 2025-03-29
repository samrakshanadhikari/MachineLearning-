import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('patientdata.csv')  # Ensure the path is correct

# Expected numeric columns (including features and regression targets)
numeric_columns = [
    'FD ROI 1 (FD-Traditional)', 'FD ROI 1 (FD-Modified)', 
    'FD ROI 2 (FD-Traditional)', 'FD ROI 2 (FD-Modified)', 
    'Cortical Thickness Pan', 'Cortical Thickness Trans', 'BMI', 
    'Age', 'Height', 'Weight', 'BMD L1-L4', 'Low BMD Spine', 
    'Spine Osteoporosis Diagnosis', 'Low BMD Hip', 'Hip Osteoporosis Diagnosis',
    'BMD Femoral Neck', 'BMD Total Femur', 'T-Score Femoral Neck', 'T-Score Total Femur'
]

# Identify which of the expected numeric columns exist in the DataFrame
existing_numeric_columns = [col for col in numeric_columns if col in df.columns]
missing_numeric_columns = set(numeric_columns) - set(existing_numeric_columns)
if missing_numeric_columns:
    print(f"Warning: The following numeric columns are missing in the dataset: {missing_numeric_columns}")

# Replace commas and convert existing numeric columns to numeric types
df[existing_numeric_columns] = df[existing_numeric_columns].replace({',': ''}, regex=True)
df[existing_numeric_columns] = df[existing_numeric_columns].apply(pd.to_numeric, errors='coerce')

# Conditionally drop rows with missing numeric data and, if present, 'Unified Diagnosis'
if 'Unified Diagnosis' in df.columns:
    df = df.dropna(subset=existing_numeric_columns + ['Unified Diagnosis'])
else:
    print("Warning: 'Unified Diagnosis' column is missing. Skipping classification step.")
    df = df.dropna(subset=existing_numeric_columns)

# Define the features used for regression.
expected_features = [
    'FD ROI 1 (FD-Traditional)', 'FD ROI 1 (FD-Modified)', 
    'FD ROI 2 (FD-Traditional)', 'FD ROI 2 (FD-Modified)', 
    'Cortical Thickness Pan', 'Cortical Thickness Trans', 'BMI', 
    'Age', 'Height', 'Weight', 'BMD L1-L4', 'Low BMD Spine', 
    'Spine Osteoporosis Diagnosis', 'Low BMD Hip', 'Hip Osteoporosis Diagnosis'
]
features = [col for col in expected_features if col in df.columns]

# Define regression targets
target_bmd_femoral_neck   = 'BMD Femoral Neck'
target_bmd_total_femur    = 'BMD Total Femur'
target_tscore_femoral_neck = 'T-Score Femoral Neck'
target_tscore_total_femur  = 'T-Score Total Femur'

# Prepare feature matrix and regression targets
X = df[features]
y_bmd_femoral_neck   = df[target_bmd_femoral_neck]
y_bmd_total_femur    = df[target_bmd_total_femur]
y_tscore_femoral_neck = df[target_tscore_femoral_neck]
y_tscore_total_femur  = df[target_tscore_total_femur]

print("Dataset shape after cleaning:", df.shape)

# Split data for regression training and testing (80% train, 20% test)
(X_train, X_test, 
 y_bmd_femoral_neck_train, y_bmd_femoral_neck_test, 
 y_bmd_total_femur_train, y_bmd_total_femur_test, 
 y_tscore_femoral_neck_train, y_tscore_femoral_neck_test, 
 y_tscore_total_femur_train, y_tscore_total_femur_test) = train_test_split(
    X, y_bmd_femoral_neck, y_bmd_total_femur, y_tscore_femoral_neck, y_tscore_total_femur,
    test_size=0.2, random_state=42)

# Initialize and train Linear Regression models
model_bmd_femoral_neck = LinearRegression()
model_bmd_total_femur = LinearRegression()
model_tscore_femoral_neck = LinearRegression()
model_tscore_total_femur = LinearRegression()

model_bmd_femoral_neck.fit(X_train, y_bmd_femoral_neck_train)
model_bmd_total_femur.fit(X_train, y_bmd_total_femur_train)
model_tscore_femoral_neck.fit(X_train, y_tscore_femoral_neck_train)
model_tscore_total_femur.fit(X_train, y_tscore_total_femur_train)

# Make predictions on the test set
y_bmd_femoral_neck_pred   = model_bmd_femoral_neck.predict(X_test)
y_bmd_total_femur_pred    = model_bmd_total_femur.predict(X_test)
y_tscore_femoral_neck_pred = model_tscore_femoral_neck.predict(X_test)
y_tscore_total_femur_pred  = model_tscore_total_femur.predict(X_test)

# Evaluate the regression models and print MSE and R^2
mse_bmd_fn = mean_squared_error(y_bmd_femoral_neck_test, y_bmd_femoral_neck_pred)
r2_bmd_fn  = r2_score(y_bmd_femoral_neck_test, y_bmd_femoral_neck_pred)
print("BMD Femoral Neck - MSE:", mse_bmd_fn, "R^2:", r2_bmd_fn)

mse_bmd_tf = mean_squared_error(y_bmd_total_femur_test, y_bmd_total_femur_pred)
r2_bmd_tf  = r2_score(y_bmd_total_femur_test, y_bmd_total_femur_pred)
print("BMD Total Femur - MSE:", mse_bmd_tf, "R^2:", r2_bmd_tf)

mse_tscore_fn = mean_squared_error(y_tscore_femoral_neck_test, y_tscore_femoral_neck_pred)
r2_tscore_fn  = r2_score(y_tscore_femoral_neck_test, y_tscore_femoral_neck_pred)
print("T-Score Femoral Neck - MSE:", mse_tscore_fn, "R^2:", r2_tscore_fn)

mse_tscore_tf = mean_squared_error(y_tscore_total_femur_test, y_tscore_total_femur_pred)
r2_tscore_tf  = r2_score(y_tscore_total_femur_test, y_tscore_total_femur_pred)
print("T-Score Total Femur - MSE:", mse_tscore_tf, "R^2:", r2_tscore_tf)

# If 'Unified Diagnosis' exists, proceed with classification tasks
if 'Unified Diagnosis' in df.columns:
    target_classification = 'Unified Diagnosis'
    # Create a DataFrame with regression predictions.
    # Set the index to match the test set so we can align with the corresponding classification target.
    X_predictions = pd.DataFrame({
        'Predicted BMD Femoral Neck': y_bmd_femoral_neck_pred,
        'Predicted BMD Total Femur': y_bmd_total_femur_pred,
        'Predicted T-Score Femoral Neck': y_tscore_femoral_neck_pred,
        'Predicted T-Score Total Femur': y_tscore_total_femur_pred
    }, index=y_bmd_femoral_neck_test.index)

    # Retrieve the corresponding classification target values from the original DataFrame
    y_classification = df.loc[X_predictions.index, target_classification]

    # Split data for classification (80% train, 20% test)
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_predictions, y_classification, test_size=0.2, random_state=42)

    # Initialize and train the Logistic Regression classifier
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_class, y_train_class)

    # Predict the classification on the test set and evaluate
    y_pred_class = logistic_model.predict(X_test_class)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    print("Classification Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test_class, y_pred_class))
    print("Confusion Matrix:\n", confusion_matrix(y_test_class, y_pred_class))
else:
    print("Skipping classification steps since 'Unified Diagnosis' column is not available.")

# Plot for BMD Total Femur: Actual vs Predicted with MSE and R^2 shown in the title
plt.figure(figsize=(10, 6))
plt.scatter(y_bmd_total_femur_test, y_bmd_total_femur_pred, color='green', alpha=0.7)
plt.plot([y_bmd_total_femur_test.min(), y_bmd_total_femur_test.max()],
         [y_bmd_total_femur_test.min(), y_bmd_total_femur_test.max()],
         color='red', linestyle='--')
plt.title(f"Actual vs Predicted BMD Total Femur\nMSE: {mse_bmd_tf:.2f}, R²: {r2_bmd_tf:.2f}")
plt.xlabel("Actual BMD Total Femur")
plt.ylabel("Predicted BMD Total Femur")
plt.show()
