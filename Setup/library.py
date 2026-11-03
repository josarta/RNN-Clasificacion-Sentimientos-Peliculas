"""
library.py
==========
Módulo de utilidades y librerías para el Miniproyecto 2:
Clasificación de Sentimientos en Reseñas de Películas (IMDb) con Redes Neuronales Recurrentes (RNN).

Contiene funciones de preprocesamiento de texto, arquitecturas recurrentes
(LSTM y GRU), entrenamiento con Early Stopping, visualización de métricas
y evaluación cuantitativa y cualitativa sobre el conjunto de prueba.

Curso: Maestría en Inteligencia Artificial / Técnicas de Deep Learning
"""

import re
import time
import copy
import string
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)


# ---------------------------------------------------------------------------
# 1. PREPROCESAMIENTO Y VECTORIZACIÓN DE TEXTO
# ---------------------------------------------------------------------------

def clean_text(text):
    """
    Limpia una cadena de texto eliminando etiquetas HTML, URLs, signos de puntuación
    y normalizando todo a minúsculas.

    Parámetros
    ----------
    text : str
        Texto original en bruto.

    Retorna
    -------
    text_limpio : str
        Texto procesado y normalizado.
    """
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_vocabulary(tokenized_texts, max_vocab_size=10000):
    """
    Construye el vocabulario a partir de las palabras más frecuentes en el corpus.

    Parámetros
    ----------
    tokenized_texts : list of list of str
        Lista de documentos previamente tokenizados.
    max_vocab_size : int, opcional
        Tamaño máximo del vocabulario (por defecto 10,000).

    Retorna
    -------
    vocab : dict
        Diccionario que mapea cada palabra a su identificador numérico único.
    word_counts : Counter
        Conteo de frecuencias de cada término.
    """
    word_counts = Counter(word for doc in tokenized_texts for word in doc)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counts.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab, word_counts


def text_to_sequence_prepad(tokens, vocab_dict, max_len=150):
    """
    Convierte una lista de tokens en un arreglo de IDs aplicando pre-padding
    (relleno a la izquierda con el token <PAD>).

    Parámetros
    ----------
    tokens : list of str
        Tokens que componen el documento.
    vocab_dict : dict
        Vocabulario generado.
    max_len : int, opcional
        Longitud fija de la secuencia temporal (por defecto 150).

    Retorna
    -------
    seq : list of int
        Secuencia numérica de tamaño exactamente igual a max_len.
    """
    seq = [vocab_dict.get(word, vocab_dict["<UNK>"]) for word in tokens[-max_len:]]
    if len(seq) < max_len:
        seq = [vocab_dict["<PAD>"]] * (max_len - len(seq)) + seq
    return seq


def prepare_data_loaders(X, y, batch_size=64, random_state=42):
    """
    Divide los datos en entrenamiento (70%), validación (15%) y prueba (15%)
    estratificados, retornando sus DataLoaders y metadatos correspondientes.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de secuencias de tamaño (muestras, timesteps).
    y : np.ndarray
        Vector de etiquetas binarias (0 o 1).
    batch_size : int, opcional
        Tamaño de lote para el entrenamiento (por defecto 64).
    random_state : int, opcional
        Semilla para asegurar reproducibilidad.

    Retorna
    -------
    train_loader, val_loader, test_loader : DataLoader
        Pipelines de datos listos para entrenamiento e inferencia.
    indices : dict
        Diccionario con los índices de cada partición (train, val, test).
    """
    indices_all = np.arange(len(y))

    # 70% entrenamiento, 30% temporal
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices_all, test_size=0.30, random_state=random_state, stratify=y
    )

    # 15% validación, 15% prueba
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.long),   torch.tensor(y_val, dtype=torch.float32))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.long),  torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    indices = {'train': idx_train, 'val': idx_val, 'test': idx_test}
    return (train_loader, val_loader, test_loader), (X_train, X_val, X_test), indices


# ---------------------------------------------------------------------------
# 2. ARQUITECTURAS NEURONALES RECURRENTES (LSTM y GRU)
# ---------------------------------------------------------------------------

class SentimentLSTM(nn.Module):
    """
    Modelo recurrente basado en LSTM para clasificación binaria de texto.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        out = self.fc(h_n[-1])
        return out.squeeze(1)


class SentimentGRU(nn.Module):
    """
    Modelo recurrente basado en GRU con regularización estocástica Dropout.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, dropout_prob=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, h_n = self.gru(emb)
        feat = self.dropout(h_n[-1])
        out = self.fc(feat)
        return out.squeeze(1)


# ---------------------------------------------------------------------------
# 3. ENTRENAMIENTO CON EARLY STOPPING
# ---------------------------------------------------------------------------

def train_sentiment_model(model, train_loader, val_loader, optimizer, criterion,
                          n_epochs=10, patience=3, device='cpu', model_name="Modelo"):
    """
    Entrena un modelo en PyTorch implementando Early Stopping y restauración
    automática de los mejores pesos.

    Retorna
    -------
    history : dict
        Evolución de 'loss', 'val_loss', 'acc', 'val_acc' por época.
    total_time : float
        Tiempo total transcurrido en segundos.
    """
    print(f"\n{'='*60}\nIniciando entrenamiento: {model_name}\n{'='*60}")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    t0 = time.time()
    for epoch in range(n_epochs):
        # Modo entrenamiento
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == yb).sum().item()
            total_train += yb.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Modo validación
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)

                running_val_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history['loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Época {epoch+1:02d}/{n_epochs:02d} — "
              f"loss: {epoch_train_loss:.4f} — acc: {epoch_train_acc*100:.2f}% | "
              f"val_loss: {epoch_val_loss:.4f} — val_acc: {epoch_val_acc*100:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping en época {epoch+1}: la pérdida de validación no mejoró en {patience} épocas.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    total_time = time.time() - t0
    print(f"Entrenamiento completado en {total_time:.2f} segundos.")
    return history, total_time


# ---------------------------------------------------------------------------
# 4. VISUALIZACIÓN DE HISTORIALES Y MATRICES
# ---------------------------------------------------------------------------

def visualization_history_comparative(hist_m1, hist_m2, name1="LSTM", name2="GRU"):
    """
    Genera dos paneles comparativos: curvas de pérdida y exactitud a lo largo de las épocas.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    # Panel 1: Pérdida
    axes[0].plot(hist_m1['loss'], color='#5a2ff4', linestyle='-', marker='.', lw=2, label=f'{name1} Train')
    axes[0].plot(hist_m1['val_loss'], color='#5a2ff4', linestyle='--', marker='.', lw=2, label=f'{name1} Val')
    axes[0].plot(hist_m2['loss'], color='#60d13e', linestyle='-', marker='.', lw=2, label=f'{name2} Train')
    axes[0].plot(hist_m2['val_loss'], color='#60d13e', linestyle='--', marker='.', lw=2, label=f'{name2} Val')
    axes[0].set_xlabel('Época', fontsize=11)
    axes[0].set_ylabel('Pérdida (BCE)', fontsize=11)
    axes[0].set_title('Evolución de la Pérdida de Entrenamiento y Validación', fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.7)
    axes[0].legend(fontsize=9, loc='upper right')

    # Panel 2: Exactitud
    axes[1].plot(hist_m1['acc'], color='#5a2ff4', linestyle='-', marker='.', lw=2, label=f'{name1} Train')
    axes[1].plot(hist_m1['val_acc'], color='#5a2ff4', linestyle='--', marker='.', lw=2, label=f'{name1} Val')
    axes[1].plot(hist_m2['acc'], color='#60d13e', linestyle='-', marker='.', lw=2, label=f'{name2} Train')
    axes[1].plot(hist_m2['val_acc'], color='#60d13e', linestyle='--', marker='.', lw=2, label=f'{name2} Val')
    axes[1].set_xlabel('Época', fontsize=11)
    axes[1].set_ylabel('Exactitud (Accuracy)', fontsize=11)
    axes[1].set_title('Evolución de la Exactitud de Clasificación', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    axes[1].legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(cm1, cm2, name1="LSTM", name2="GRU", acc1=0.0, acc2=0.0):
    """
    Visualiza las matrices de confusión para dos modelos lado a lado.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=['Negativo (0)', 'Positivo (1)'])
    disp1.plot(ax=axes[0], cmap='Purples', colorbar=False)
    axes[0].set_title(f"{name1} (Exactitud: {acc1*100:.1f}%)", fontsize=11)
    axes[0].set_xlabel('Etiqueta Predicha', fontsize=10)
    axes[0].set_ylabel('Etiqueta Real (Ground Truth)', fontsize=10)

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['Negativo (0)', 'Positivo (1)'])
    disp2.plot(ax=axes[1], cmap='Greens', colorbar=False)
    axes[1].set_title(f"{name2} (Exactitud: {acc2*100:.1f}%)", fontsize=11)
    axes[1].set_xlabel('Etiqueta Predicha', fontsize=10)
    axes[1].set_ylabel('Etiqueta Real (Ground Truth)', fontsize=10)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. EVALUACIÓN EN CONJUNTO DE PRUEBA
# ---------------------------------------------------------------------------

def evaluate_test_set(model, loader, device='cpu'):
    """
    Genera inferencia en el conjunto de prueba y calcula el compendio
    completo de métricas estadísticas.
    """
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    return {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds),
        'recall': recall_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds),
        'auc': roc_auc_score(all_targets, all_probs),
        'cm': confusion_matrix(all_targets, all_preds),
        'probs': all_probs,
        'preds': all_preds,
        'targets': all_targets
    }

