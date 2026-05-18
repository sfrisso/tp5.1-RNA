import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim

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

# Completar faltantes
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Escalado
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Balanceo
# -----------------------------

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# -----------------------------
# División train/test
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Tensor PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# -----------------------------
# Modelo RNA
# -----------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(X_train.shape[1], 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = MLP()

# -----------------------------
# Entrenamiento
# -----------------------------

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# -----------------------------
# Evaluación
# -----------------------------

with torch.no_grad():

    predictions = model(X_test)

    predictions = (predictions >= 0.5).float()

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))