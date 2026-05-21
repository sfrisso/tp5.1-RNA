# -------------------
# Evaluación del modelo
# Este código se encarga de evaluar el rendimiento del modelo en el conjunto de prueba.
# Utiliza métricas como la precisión, el reporte de clasificación y la matriz de confusión
# para analizar el desempeño del modelo en la tarea de clasificación.
# -------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import torch

# Importamos los modelos
def evaluate_model(model, X_test_tensor, y_test):
    # Ponemos el modelo en modo evaluación
    model.eval()
    # Desactivamos el cálculo de gradientes para la evaluación
    with torch.no_grad():
        # Obtenemos las predicciones del modelo
        predictions = model(X_test_tensor)
        # Convertimos las predicciones a etiquetas binarias (0 o 1) usando un umbral de 0.5
        predictions = (predictions >= 0.5).float()
    # Convertimos las predicciones a un formato compatible con sklearn (numpy array)
    y_pred = predictions.numpy()
    # Calculamos la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Imprimimos los resultados de la evaluación
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }