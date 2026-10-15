
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/best_model.joblib")

# Features used by the Boston Housing dataset
features = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

print("\n=== Boston House Price Prediction ===")
print("Enter the values for the 13 housing features.\n")

values = {}


try:
    for feature in features:
        values[feature] = float(input(f"Enter {feature}: "))

    # Basic input validation
    if any(not pd.notna(v) for v in values.values()):
        raise ValueError("Values cannot be NaN.")

    if any(v < 0 for v in values.values()):
        raise ValueError("Feature values cannot be negative.")

    if values["CHAS"] not in (0, 1):
        raise ValueError("CHAS must be 0 or 1.")

    input_data = pd.DataFrame([values], columns=features)
    prediction = model.predict(input_data)[0]

    print("\n" + "=" * 40)
    print("      BOSTON HOUSE PRICE RESULT")
    print("=" * 40)
    print(f"Predicted price: ${prediction * 1000:,.2f}")
    print(f"Price in thousands: {prediction:.2f}")
    print("=" * 40)

except ValueError as error:
    print("Invalid input:", error)
