
import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the provided dataset
files = glob.glob("data/*.csv") + glob.glob("data/*.xlsx") + glob.glob("data/*.xls")

if not files:
    raise FileNotFoundError(
        "Place your CSV or Excel dataset inside the data folder."
    )

file = files[0]
df = pd.read_excel(file) if file.endswith((".xlsx", ".xls")) else pd.read_csv(file)

print("Dataset loaded:", file)
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# 2. Identify the target column
# Change TARGET_COLUMN if automatic detection chooses incorrectly.
TARGET_COLUMN = None

target_candidates = [
    "medv", "price", "house_price", "house price",
    "target", "median_house_value", "median_house_price"
]

if TARGET_COLUMN is None:
    matches = {
        str(col).strip().lower(): col for col in df.columns
    }
    for candidate in target_candidates:
        if candidate in matches:
            TARGET_COLUMN = matches[candidate]
            break

if TARGET_COLUMN is None:
    raise ValueError(
        "Target column not detected. Set TARGET_COLUMN to the "
        "actual house-price column name shown above."
    )

print("\nTarget column:", TARGET_COLUMN)

# 3. Clean target and prepare features
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)
df = df.dropna(subset=[TARGET_COLUMN])

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Remove identifier columns if present (edit if needed)
id_columns = [
    col for col in X.columns
    if str(col).strip().lower() in ["id", "index", "unnamed: 0"]
]
X = X.drop(columns=id_columns)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

# 4. Preprocess missing values and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. Select regression models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []
best_model = None
best_name = None
best_rmse = float("inf")
best_predictions = None

os.makedirs("models", exist_ok=True)

# 7. Train and evaluate
for name, regressor in models.items():
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", regressor)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"\n{name}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R2 Score : {r2:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name
        best_predictions = predictions

# 8. Compare models
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
results_df.to_csv("model_results.csv", index=False)

# 9. Save best model
joblib.dump(best_model, "models/best_model.joblib")
print(f"\nBest model: {best_name}")
print("Model saved to models/best_model.joblib")



# 10. Feature importance for tree-based models
if best_name in ["Random Forest", "Gradient Boosting"]:

    regressor = best_model.named_steps["regressor"]
    preprocessor = best_model.named_steps["preprocessing"]

    importance = regressor.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    print("\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))

    top_features = importance_df.head(10).sort_values("Importance")

    plt.figure(figsize=(8, 5))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()


# 11. Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_predictions, alpha=0.7)

low = min(y_test.min(), best_predictions.min())
high = max(y_test.max(), best_predictions.max())

plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title(f"Actual vs Predicted: {best_name}")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# 12. Predict a sample test record
sample = X_test.iloc[[0]]
prediction = best_model.predict(sample)[0]

print("\nSample prediction")
print("Actual price:", y_test.iloc[0])
print("Predicted price:", round(prediction, 3))
print("\nProject completed successfully!")
