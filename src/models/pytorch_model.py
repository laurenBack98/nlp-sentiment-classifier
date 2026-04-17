"""
models/pytorch_model.py
Red neuronal densa (fully connected) para clasificación de texto con PyTorch.
Incluye Early Stopping y vectorización TF-IDF.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =============================================================================
#  ARQUITECTURA
# =============================================================================

class TextClassifier(nn.Module):
    """
    Red neuronal densa con 2 capas ocultas, Batch Normalization y Dropout.

    Arquitectura
    ------------
    Linear → BN → ReLU → Dropout →
    Linear → BN → ReLU → Dropout →
    Linear (salida)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
#  EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Detiene el entrenamiento si val_loss no mejora durante `patience` épocas.
    """

    def __init__(self, patience: int = 5):
        self.patience  = patience
        self.best_loss = float('inf')
        self.counter   = 0
        self.stop      = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: sin mejora {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True


# =============================================================================
#  PREPARACIÓN DE DATOS
# =============================================================================

def prepare_tfidf_tensors(
    features: pd.DataFrame,
    text_col: str = 'token',
    target_col: str = 'sentiment_map',
    max_features: int = 5_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Vectoriza con TF-IDF y convierte a tensores PyTorch.

    Retorna
    -------
    X_tr_t, X_val_t, y_tr_t, y_val_t, y_val (numpy), tfidf
    """
    tfidf = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    X_vec = tfidf.fit_transform(features[text_col]).toarray()
    y_vec = features[target_col].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_vec, y_vec, test_size=test_size, random_state=random_state, stratify=y_vec
    )

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    return X_tr_t, X_val_t, y_tr_t, y_val_t, y_val, tfidf


# =============================================================================
#  ENTRENAMIENTO
# =============================================================================

def train_pytorch_model(
    features: pd.DataFrame,
    text_col: str = 'token',
    target_col: str = 'sentiment_map',
    max_features: int = 5_000,
    hidden_dim: int = 256,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 5,
    lr: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Pipeline completo: prepara datos, entrena y evalúa el TextClassifier.

    Retorna
    -------
    model   : modelo entrenado
    y_val   : etiquetas verdaderas de validación
    preds   : predicciones de validación
    """
    # Datos
    X_tr_t, X_val_t, y_tr_t, y_val_t, y_val, _ = prepare_tfidf_tensors(
        features, text_col, target_col, max_features, test_size, random_state
    )

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size)

    # Modelo
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = TextClassifier(input_dim=X_tr_t.shape[1], hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    es        = EarlyStopping(patience=patience)

    # Loop de entrenamiento
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = sum(
            _train_step(model, optimizer, criterion, xb.to(device), yb.to(device))
            for xb, yb in train_loader
        )

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb  = xb.to(device), yb.to(device)
                out      = model(xb)
                val_loss += criterion(out, yb).item()
                correct  += (out.argmax(1) == yb).sum().item()

        val_loss /= len(val_loader)
        val_acc   = correct / len(X_val_t)
        print(f"Época {epoch:3d} | train_loss: {train_loss/len(train_loader):.4f}"
              f" | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        es.step(val_loss)
        if es.stop:
            print(f"\n⏹ Early Stopping activado en época {epoch}.")
            break

    # Evaluación final
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t.to(device)).argmax(1).cpu().numpy()

    print(f"\nAccuracy PyTorch: {accuracy_score(y_val, preds):.4f}")
    print(classification_report(y_val, preds, target_names=['Negativo', 'Positivo']))

    return model, y_val, preds


def _train_step(model, optimizer, criterion, xb, yb) -> float:
    optimizer.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()
    optimizer.step()
    return loss.item()
