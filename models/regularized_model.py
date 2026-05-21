import torch
import torch.nn as nn

# -----------------------------
# Modelo perceptrón multicapa (MLP) con regularización
# -----------------------------
class RegularizedModel(nn.Module):
    def __init__(self, input_size):
        # Llamamos al constructor de la clase base (nn.Module)
        super(RegularizedModel, self).__init__()

        # Definimos la arquitectura de la red neuronal con regularización (BatchNorm + Dropout)
        self.network = nn.Sequential(
            # Capa de entrada a oculta
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64), # Normalización por lotes para estabilizar el entrenamiento
            nn.ReLU(), # Función de activación ReLU para introducir no linealidad
            nn.Dropout(0.3), # Dropout para reducir el sobreajuste (30% de las neuronas se apagarán aleatoriamente durante el entrenamiento)

            # Capa oculta a oculta
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Capa oculta a oculta
            nn.Linear(32, 16),
            nn.ReLU(),

            # Capa oculta a salida
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    # Definimos el método forward para la propagación hacia adelante
    def forward(self, x):
        return self.network(x)