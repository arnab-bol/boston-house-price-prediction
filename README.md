# Boston House Price Prediction

## Project Overview
This project develops a machine learning regression model to predict Boston housing prices using housing-related features such as crime rate, average number of rooms, property tax rate, and lower-status population percentage.

Developed as part of the **ShadowFox Machine Learning Internship**.

## Objectives
- Preprocess the provided housing dataset.
- Handle missing numerical values using median imputation.
- Train and compare multiple regression algorithms.
- Evaluate performance using MAE, RMSE, and R².
- Save the best-performing model for future predictions.

## Dataset
- Dataset file: `HousingData.csv`
- Records: 506
- Features: 13
- Target variable: `MEDV`

The target represents median owner-occupied home value, expressed in thousands of dollars in the original Boston Housing dataset.

## Algorithms Used
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

## Model Evaluation

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 3.1476 | 4.9983 | 0.6593 |
| Random Forest | 2.0668 | 2.9668 | 0.8800 |
| Gradient Boosting | 1.9224 | 2.7087 | 0.8999 |

Gradient Boosting achieved the lowest RMSE among the three evaluated models.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Project Structure

```text
boston-house-price-prediction/
├── data/
│   └── HousingData.csv
├── models/
│   └── best_model.joblib
├── main.py
├── predict.py
├── model_results.csv
├── actual_vs_predicted.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Execution

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the models:

```bash
python main.py
```

Run the prediction program:

```bash
python predict.py
```

## Conclusion
The project demonstrates the complete regression workflow, including preprocessing, training, model comparison, evaluation, visualization, and saving a trained model for prediction.

## Limitation
The historical Boston Housing dataset has known ethical limitations and should be used for educational purposes rather than real-world housing decisions.