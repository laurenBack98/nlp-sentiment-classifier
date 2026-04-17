"""
models/rnn_models.py
Modelos de redes neuronales recurrentes con TensorFlow/Keras:
RNN simple, LSTM y GRU.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
)


# =============================================================================
#  PREPARACIÓN DE SECUENCIAS
# =============================================================================

def prepare_sequences(
    df: pd.DataFrame,
    text_col: str = 'review_lematizer_str',
    target_col: str = 'sentiment_map',
    max_words: int = 10_000,
    max_len: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Tokeniza el texto con Keras Tokenizer, aplica padding y hace el split.

    Retorna
    -------
    X_train, X_test, y_train, y_test, tokenizer
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df[text_col])

    sequences = tokenizer.texts_to_sequences(df[text_col])
    X = pad_sequences(sequences, maxlen=max_len, padding='post')
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, tokenizer


# =============================================================================
#  ARQUITECTURAS
# =============================================================================

def build_rnn(max_words: int = 10_000, max_len: int = 100,
              embed_dim: int = 64, units: int = 64,
              dropout: float = 0.5) -> Sequential:
    """Red Neuronal Recurrente simple."""
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_len),
        SimpleRNN(units),
        Dropout(dropout),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_lstm(max_words: int = 10_000, max_len: int = 100,
               embed_dim: int = 64, units: int = 128,
               dropout: float = 0.5) -> Sequential:
    """LSTM - Memoria a Largo y Corto Plazo."""
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_len),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_gru(max_words: int = 10_000, max_len: int = 100,
              embed_dim: int = 64, units: int = 128,
              dropout: float = 0.5) -> Sequential:
    """GRU - Unidad Recurrente con Compuertas."""
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_len),
        GRU(units, return_sequences=False),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# =============================================================================
#  ENTRENAMIENTO GENÉRICO
# =============================================================================

def train_rnn_model(
    model: Sequential,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 5,
    batch_size: int = 32,
) -> tuple:
    """
    Entrena un modelo Keras y lo evalúa en el set de test.

    Retorna
    -------
    history : historial de entrenamiento (para plot_history)
    loss    : pérdida en test
    accuracy: exactitud en test
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss:.4f}  |  Accuracy: {accuracy:.4f}")
    return history, loss, accuracy
