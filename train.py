# -------------------
# Entrenamiento del modelo
# Este código se encarga de entrenar el modelo de red neuronal utilizando el conjunto de entrenamiento.
# Utiliza una función de pérdida (criterio) y un optimizador para ajustar los pesos del modelo durante el proceso de entrenamiento.
# -------------------

import torch
import time

# Importamos los modelos
def train_model(model, criterion, optimizer,
                X_train_tensor, y_train_tensor,
                epochs=100):
    
    # Lista para almacenar las pérdidas durante el entrenamiento
    losses = []
    start_time = time.time()

    # Iteramos sobre el número de épocas para entrenar el modelo
    for epoch in range(epochs):
        #model.train() #Activa dropout.
        # Ponemos el modelo en modo entrenamiento
        outputs = model(X_train_tensor)
        # Calculamos la pérdida entre las predicciones del modelo y las etiquetas reales
        loss = criterion(outputs, y_train_tensor)
        # Reiniciamos los gradientes del optimizador para evitar acumulación de gradientes de épocas anteriores
        optimizer.zero_grad()
        # Calculamos los gradientes de la pérdida con respecto a los pesos del modelo
        loss.backward()
        # Actualizamos los pesos del modelo utilizando el optimizador
        optimizer.step()
        # Almacenamos la pérdida actual en la lista de pérdidas
        losses.append(loss.item())
        # Imprimimos la pérdida cada 10 épocas para monitorear el progreso del entrenamiento
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Loss: {loss.item():.4f}")
    end_time = time.time()
    print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")
    return losses