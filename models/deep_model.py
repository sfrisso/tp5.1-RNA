import torch
import torch.nn as nn

# -----------------------------
# Modelo perceptrón multicapa (MLP) profundo
# -----------------------------
class DeepModel(nn.Module):
    def __init__(self, input_size):
        # Llamamos al constructor de la clase base (nn.Module)
        super(DeepModel, self).__init__()

        # Definimos la arquitectura de la red neuronal
        self.network = nn.Sequential(
            # Capa de entrada a oculta
            nn.Linear(input_size, 64),
            nn.ReLU(),
            # Capa oculta a oculta
            nn.Linear(64, 32),
            nn.ReLU(),
            # Capa oculta a oculta
            nn.Linear(32, 16),
            nn.ReLU(),
            # Capa oculta a oculta
            nn.Linear(16, 8),
            nn.ReLU(),
            # Capa oculta a salida
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    # Definimos el método forward para la propagación hacia adelante
    def forward(self, x):
        return self.network(x)