# -----------------------------
# Main.py
# Este código es el punto de entrada del proyecto. Se encarga de cargar el dataset, 
# preprocesar los datos, definir los modelos, entrenarlos y evaluarlos.
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from models.base_model import BaseModel
from models.deep_model import DeepModel
from models.regularized_model import RegularizedModel

import pandas as pd
import time

from train import train_model
from evaluate import evaluate_model

# -----------------------------
# Cargar dataset
# -----------------------------

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# -----------------------------
# Preprocesamiento
# -----------------------------

# Eliminar columnas irrelevantes
# Se elimina cabin porque tiene muchos datos faltantes
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# Variables categóricas
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Separar X e y
X = df.drop("Survived", axis=1)
y = df["Survived"]

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Imputación (SOLO train fit)
# -----------------------------
imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# -----------------------------
# SMOTE (solo train)
# -----------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -----------------------------
# Escalado (solo train fit)
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Tensores
# -----------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# -----------------------------
# MODELOS
# -----------------------------

input_size = X_train.shape[1]

models = {
    "BaseModel": BaseModel(input_size),
    "DeepModel": DeepModel(input_size),
    "RegularizedModel": RegularizedModel(input_size)
}

results = []

# -----------------------------
# ENTRENAMIENTO
# -----------------------------

for name, model in models.items():

    print(f"\n========== {name} ==========")
    start_time = time.time()

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    train_model(
        model,
        criterion,
        optimizer,
        X_train_tensor,
        y_train_tensor,
        epochs=100
    )

    metrics = evaluate_model(
        model,
        X_test_tensor,
        y_test
    )

    end_time = time.time()

    execution_time = end_time - start_time

    results.append({
        "Modelo": name,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-Score": metrics["f1"],
        "Tiempo (s)": execution_time
    })

# -----------------------------
# RESULTADOS FINALES
# -----------------------------

print("\n===== RESULTADOS =====")

results_df = pd.DataFrame(results)

print(results_df)

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\n===== MEJOR MODELO =====")
print(best_model)