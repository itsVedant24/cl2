
# Install required libraries:
# pip install pandas numpy pgmpy

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# 1. Load Heart Disease Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

data = pd.read_csv(url, names=columns)

# 2. Remove missing values
data = data.replace("?", pd.NA)
data = data.dropna()

# 3. Convert numerical values into categories
data["age"] = pd.cut(
    data["age"].astype(float),
    bins=[0, 45, 55, 100],
    labels=["young", "middle", "senior"]
)

data["trestbps"] = pd.cut(
    data["trestbps"].astype(float),
    bins=[0, 120, 140, 300],
    labels=["normal", "high", "very_high"]
)

data["chol"] = pd.cut(
    data["chol"].astype(float),
    bins=[0, 200, 240, 600],
    labels=["normal", "borderline", "high"]
)

data["thalach"] = pd.cut(
    data["thalach"].astype(float),
    bins=[0, 120, 150, 250],
    labels=["low", "medium", "high"]
)

data["oldpeak"] = pd.cut(
    data["oldpeak"].astype(float),
    bins=[-1, 1, 2, 10],
    labels=["low", "medium", "high"]
)

# 4. Convert target into yes or no
data["target"] = data["target"].astype(int)

data["target"] = data["target"].apply(
    lambda x: "yes" if x > 0 else "no"
)

# Convert all values into strings
data = data.astype(str)

# 5. Create Naive Bayes Network
model = DiscreteBayesianNetwork([
    ("target", "age"),
    ("target", "sex"),
    ("target", "cp"),
    ("target", "trestbps"),
    ("target", "chol"),
    ("target", "thalach"),
    ("target", "exang"),
    ("target", "oldpeak"),
    ("target", "ca"),
    ("target", "thal")
])

# 6. Train the Bayesian Network
estimator = BayesianEstimator(model, data)

cpds = estimator.get_parameters(
    prior_type="BDeu"
)

model.add_cpds(*cpds)

# 7. Check model
print("Bayesian Network constructed successfully!")
print("Model is valid:", model.check_model())

# 8. Perform diagnosis
inference = VariableElimination(model)

# Patient details
patient = {
    "age": "senior",
    "sex": "1.0",
    "cp": "4.0",
    "trestbps": "very_high",
    "chol": "high",
    "thalach": "low",
    "exang": "1.0",
    "oldpeak": "high",
    "ca": "3.0",
    "thal": "7.0"
}

# 9. Calculate probability
result = inference.query(
    variables=["target"],
    evidence=patient
)

# 10. Display diagnosis
no_probability = result.get_value(target="no") * 100
yes_probability = result.get_value(target="yes") * 100

print("\nPatient Diagnosis:")
print("No Heart Disease :", round(no_probability, 2), "%")
print("Heart Disease    :", round(yes_probability, 2), "%")

print("\nFinal Result:")

if yes_probability > no_probability:
    print("Prediction: Heart Disease Detected")
elif no_probability > yes_probability:
    print("Prediction: No Heart Disease Detected")
else:
    print("Prediction: Uncertain - Equal Probability")




# 3. Heart Disease Diagnosis Using Bayesian Network
# A Bayesian Network is a probabilistic model that represents relationships between different 
# variables using a Directed Acyclic Graph (DAG). In this practical, the UCI Heart Disease Dataset 
# is used to predict heart disease based on patient details such as age, cholesterol, blood pressure 
# and chest pain. The pgmpy library is used to construct the network, calculate probabilities and perform 
# diagnosis using Bayesian inference.