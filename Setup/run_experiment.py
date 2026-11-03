"""
run_experiment.py
=================
Script de experimentación para el Microproyecto de RNN:
Clasificación de Sentimientos en Reseñas de Películas (IMDb).

Ejecuta el preprocesamiento, entrenamiento comparativo (LSTM vs GRU),
cálculo riguroso de métricas cuantitativas, análisis cualitativo y generación
de figuras en alta resolución para el informe IEEE y el notebook.
"""

import os
import re
import time
import copy
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
)

# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y REPRODUCIBILIDAD
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Dispositivo utilizado: {device}")

OUTPUT_DIR = r"D:\Maestria\Deep Learning\MiniProyecto2"
DATASET_PATH = os.path.join(OUTPUT_DIR, "IMDB-Dataset.csv")

# ---------------------------------------------------------------------------
# 2. CARGA Y PREPROCESAMIENTO DE TEXTO
# ---------------------------------------------------------------------------
print("[*] Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

# Muestreo estratificado de 20,000 reseñas para experimentación eficiente en CPU
# garantizando balance perfecto (10,000 positivas, 10,000 negativas)
df_sample = df.groupby('sentiment', group_keys=False).apply(
    lambda x: x.sample(n=10000, random_state=RANDOM_SEED)
).reset_index(drop=True)

print(f"[*] Muestras seleccionadas: {len(df_sample)} (50% positivas, 50% negativas)")

def clean_text(text):
    """
    Limpia el texto eliminando etiquetas HTML, puntuación y convirtiendo a minúsculas.
    """
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)               # Eliminar tags HTML
    text = re.sub(r'https?://\S+|www\.\S+', '', text)     # Eliminar URLs
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text) # Remover puntuación
    text = re.sub(r'\s+', ' ', text).strip()             # Espacios múltiples
    return text

print("[*] Limpiando texto...")
df_sample['cleaned_text'] = df_sample['review'].apply(clean_text)
df_sample['target'] = df_sample['sentiment'].map({'positive': 1, 'negative': 0})

# Tokenización y construcción del vocabulario
print("[*] Construyendo vocabulario...")
from collections import Counter

tokenized_texts = [text.split() for text in df_sample['cleaned_text']]
word_counts = Counter(word for doc in tokenized_texts for word in doc)

VOCAB_SIZE = 10000  # Las 10,000 palabras más frecuentes
MAX_SEQ_LEN = 150   # Longitud máxima de secuencia

# Índices especiales: 0 -> <PAD>, 1 -> <UNK>
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.most_common(VOCAB_SIZE - 2):
    vocab[word] = len(vocab)

def text_to_sequence(tokens, vocab, max_len):
    """
    Convierte lista de tokens a vector numérico de tamaño fijo (truncando o rellenando).
    """
    seq = [vocab.get(word, vocab["<UNK>"]) for word in tokens[:max_len]]
    if len(seq) < max_len:
        seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
    return seq

X_all = np.array([text_to_sequence(tokens, vocab, MAX_SEQ_LEN) for tokens in tokenized_texts])
y_all = df_sample['target'].values

# División Train (70%), Val (15%), Test (15%) estratificada
X_train_raw, X_temp, y_train, y_temp, indices_train, indices_temp = train_test_split(
    X_all, y_all, np.arange(len(df_sample)),
    test_size=0.30, random_state=RANDOM_SEED, stratify=y_all
)

X_val_raw, X_test_raw, y_val, y_test, indices_val, indices_test = train_test_split(
    X_temp, y_temp, indices_temp,
    test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"[*] Train set: {X_train_raw.shape[0]} muestras")
print(f"[*] Val set:   {X_val_raw.shape[0]} muestras")
print(f"[*] Test set:  {X_test_raw.shape[0]} muestras")

# DataLoaders de PyTorch
BATCH_SIZE = 64
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train_raw, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val_raw, dtype=torch.long), torch.tensor(y_val, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test_raw, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=False
)

# ---------------------------------------------------------------------------
# 3. DEFINICIÓN DE MODELOS (LSTM vs GRU)
# ---------------------------------------------------------------------------

class SentimentLSTM(nn.Module):
    """
    Modelo recurrente LSTM estándar para clasificación de secuencias de texto.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x)                 # [batch_size, seq_len, embed_dim]
        _, (h_n, _) = self.lstm(emb)           # h_n: [1, batch_size, hidden_dim]
        out = self.fc(h_n[-1])                  # [batch_size, 1]
        return out.squeeze(1)


class SentimentGRU(nn.Module):
    """
    Modelo recurrente GRU con capa de regularización Dropout.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout_prob=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, h_n = self.gru(emb)                  # h_n: [1, batch_size, hidden_dim]
        feat = self.dropout(h_n[-1])
        out = self.fc(feat)
        return out.squeeze(1)


# ---------------------------------------------------------------------------
# 4. FUNCIÓN DE ENTRENAMIENTO CON EARLY STOPPING
# ---------------------------------------------------------------------------

def train_sentiment_model(model, train_loader, val_loader, optimizer, criterion,
                          n_epochs=15, patience=4, model_name="Model"):
    print(f"\n{'='*50}\nIniciando entrenamiento: {model_name}\n{'='*50}")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    start_time = time.time()
    for epoch in range(n_epochs):
        # Modo Entrenamiento
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == yb).sum().item()
            total_train += yb.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Modo Validación
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history['loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Época {epoch+1:02d}/{n_epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[!] Early stopping activado en época {epoch+1}. Restaurando mejores pesos.")
                break

    total_time = time.time() - start_time
    print(f"[*] Entrenamiento completado en {total_time:.2f} s")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return history, total_time


# ---------------------------------------------------------------------------
# 5. EJECUCIÓN EXPERIMENTAL DE AMBOS MODELOS
# ---------------------------------------------------------------------------
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 0.001
N_EPOCHS = 12

criterion = nn.BCEWithLogitsLoss()

# Modelo 1: LSTM
model_lstm = SentimentLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)
params_lstm = sum(p.numel() for p in model_lstm.parameters() if p.requires_grad)
opt_lstm = torch.optim.Adam(model_lstm.parameters(), lr=LR)
hist_lstm, time_lstm = train_sentiment_model(
    model_lstm, train_loader, val_loader, opt_lstm, criterion,
    n_epochs=N_EPOCHS, patience=3, model_name="Modelo 1: LSTM Estándar"
)

# Modelo 2: GRU con Dropout
model_gru = SentimentGRU(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, dropout_prob=0.3).to(device)
params_gru = sum(p.numel() for p in model_gru.parameters() if p.requires_grad)
opt_gru = torch.optim.Adam(model_gru.parameters(), lr=LR)
hist_gru, time_gru = train_sentiment_model(
    model_gru, train_loader, val_loader, opt_gru, criterion,
    n_epochs=N_EPOCHS, patience=3, model_name="Modelo 2: GRU + Dropout"
)

# ---------------------------------------------------------------------------
# 6. EVALUACIÓN DETALLADA EN CONJUNTO DE PRUEBA
# ---------------------------------------------------------------------------

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
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

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds)
    rec = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': auc, 'cm': cm, 'probs': all_probs, 'preds': all_preds
    }

eval_lstm = evaluate_model(model_lstm, test_loader)
eval_gru = evaluate_model(model_gru, test_loader)

print("\n" + "="*50)
print("RESULTADOS FINALES EN TEST SET (3,000 MUESTRAS)")
print("="*50)
print(f"Métricas LSTM | Acc: {eval_lstm['accuracy']*100:.2f}% | Prec: {eval_lstm['precision']*100:.2f}% | Rec: {eval_lstm['recall']*100:.2f}% | F1: {eval_lstm['f1']*100:.2f}% | AUC: {eval_lstm['auc']:.4f}")
print(f"Métricas GRU  | Acc: {eval_gru['accuracy']*100:.2f}% | Prec: {eval_gru['precision']*100:.2f}% | Rec: {eval_gru['recall']*100:.2f}% | F1: {eval_gru['f1']*100:.2f}% | AUC: {eval_gru['auc']:.4f}")

# ---------------------------------------------------------------------------
# 7. GENERACIÓN DE FIGURAS PARA EL INFORME IEEE
# ---------------------------------------------------------------------------

# Figura 1: Curvas de aprendizaje comparativas (Loss y Accuracy)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(hist_lstm['loss'], label='LSTM Train', color='#5a2ff4', linestyle='-', lw=1.8)
axes[0].plot(hist_lstm['val_loss'], label='LSTM Val', color='#5a2ff4', linestyle='--', lw=1.8)
axes[0].plot(hist_gru['loss'], label='GRU Train', color='#00a86b', linestyle='-', lw=1.8)
axes[0].plot(hist_gru['val_loss'], label='GRU Val', color='#00a86b', linestyle='--', lw=1.8)
axes[0].set_title('(a) Pérdida de Entrenamiento y Validación (BCE)', fontsize=11)
axes[0].set_xlabel('Época', fontsize=10)
axes[0].set_ylabel('Pérdida', fontsize=10)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(fontsize=9)

axes[1].plot(hist_lstm['acc'], label='LSTM Train', color='#5a2ff4', linestyle='-', lw=1.8)
axes[1].plot(hist_lstm['val_acc'], label='LSTM Val', color='#5a2ff4', linestyle='--', lw=1.8)
axes[1].plot(hist_gru['acc'], label='GRU Train', color='#00a86b', linestyle='-', lw=1.8)
axes[1].plot(hist_gru['val_acc'], label='GRU Val', color='#00a86b', linestyle='--', lw=1.8)
axes[1].set_title('(b) Exactitud de Clasificación (Accuracy)', fontsize=11)
axes[1].set_xlabel('Época', fontsize=10)
axes[1].set_ylabel('Exactitud', fontsize=10)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(fontsize=9)

plt.tight_layout()
fig_loss_path = os.path.join(OUTPUT_DIR, "fig_curvas_aprendizaje.png")
plt.savefig(fig_loss_path, dpi=300)
plt.close()
print(f"[*] Figura guardada: {fig_loss_path}")

# Figura 2: Matrices de confusión comparativas
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

disp_lstm = ConfusionMatrixDisplay(confusion_matrix=eval_lstm['cm'], display_labels=['Negativo', 'Positivo'])
disp_lstm.plot(ax=axes[0], cmap='Purples', colorbar=False)
axes[0].set_title(f'LSTM (Acc: {eval_lstm["accuracy"]*100:.1f}%)', fontsize=11)
axes[0].set_xlabel('Predicción', fontsize=10)
axes[0].set_ylabel('Etiqueta Real', fontsize=10)

disp_gru = ConfusionMatrixDisplay(confusion_matrix=eval_gru['cm'], display_labels=['Negativo', 'Positivo'])
disp_gru.plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title(f'GRU + Dropout (Acc: {eval_gru["accuracy"]*100:.1f}%)', fontsize=11)
axes[1].set_xlabel('Predicción', fontsize=10)
axes[1].set_ylabel('Etiqueta Real', fontsize=10)

plt.tight_layout()
fig_cm_path = os.path.join(OUTPUT_DIR, "fig_matriz_confusion.png")
plt.savefig(fig_cm_path, dpi=300)
plt.close()
print(f"[*] Figura guardada: {fig_cm_path}")

# ---------------------------------------------------------------------------
# 8. ANÁLISIS CUALITATIVO (ACIERTOS Y FALLOS)
# ---------------------------------------------------------------------------
test_df = df_sample.iloc[indices_test].copy().reset_index(drop=True)
test_df['lstm_pred'] = eval_lstm['preds']
test_df['lstm_prob'] = eval_lstm['probs']
test_df['gru_pred'] = eval_gru['preds']
test_df['gru_prob'] = eval_gru['probs']

# Filtrar ejemplos característicos
# Caso 1: Acierto Positivo (Verdadero Positivo claro)
tp_sample = test_df[(test_df['target'] == 1) & (test_df['gru_pred'] == 1) & (test_df['gru_prob'] > 0.90)].iloc[0]

# Caso 2: Acierto Negativo (Verdadero Negativo claro)
tn_sample = test_df[(test_df['target'] == 0) & (test_df['gru_pred'] == 0) & (test_df['gru_prob'] < 0.10)].iloc[0]

# Caso 3: Falso Positivo (Real Negativo pero clasificado Positivo - ej: sarcasmo o elogio técnico con juicio desfavorable)
fp_sample = test_df[(test_df['target'] == 0) & (test_df['gru_pred'] == 1)].iloc[0]

# Caso 4: Falso Negativo (Real Positivo pero clasificado Negativo - ej: película de terror o temas oscuros bien elogiados)
fn_sample = test_df[(test_df['target'] == 1) & (test_df['gru_pred'] == 0)].iloc[0]

print("\n--- CASOS CUALITATIVOS DESTACADOS ---")
print(f"[TP - Acierto Positivo] Prob: {tp_sample['gru_prob']:.3f} | Real: Positivo")
print(f"Texto: {tp_sample['review'][:200]}...")

print(f"\n[TN - Acierto Negativo] Prob: {tn_sample['gru_prob']:.3f} | Real: Negativo")
print(f"Texto: {tn_sample['review'][:200]}...")

print(f"\n[FP - Error Falso Positivo] Prob: {fp_sample['gru_prob']:.3f} | Real: Negativo | Pred: Positivo")
print(f"Texto: {fp_sample['review'][:200]}...")

print(f"\n[FN - Error Falso Negativo] Prob: {fn_sample['gru_prob']:.3f} | Real: Positivo | Pred: Negativo")
print(f"Texto: {fn_sample['review'][:200]}...")

# Guardar métricas en archivo JSON para usar fielmente en LaTeX y el notebook
import json
results_dict = {
    'params_lstm': params_lstm,
    'params_gru': params_gru,
    'time_lstm': round(time_lstm, 2),
    'time_gru': round(time_gru, 2),
    'lstm': {
        'accuracy': round(float(eval_lstm['accuracy']) * 100, 2),
        'precision': round(float(eval_lstm['precision']) * 100, 2),
        'recall': round(float(eval_lstm['recall']) * 100, 2),
        'f1': round(float(eval_lstm['f1']) * 100, 2),
        'auc': round(float(eval_lstm['auc']), 4),
        'cm': eval_lstm['cm'].tolist()
    },
    'gru': {
        'accuracy': round(float(eval_gru['accuracy']) * 100, 2),
        'precision': round(float(eval_gru['precision']) * 100, 2),
        'recall': round(float(eval_gru['recall']) * 100, 2),
        'f1': round(float(eval_gru['f1']) * 100, 2),
        'auc': round(float(eval_gru['auc']), 4),
        'cm': eval_gru['cm'].tolist()
    },
    'qualitative_samples': {
        'tp': {'text': tp_sample['review'][:240], 'target': 'Positivo', 'prob': round(float(tp_sample['gru_prob']), 3), 'pred': 'Positivo'},
        'tn': {'text': tn_sample['review'][:240], 'target': 'Negativo', 'prob': round(float(tn_sample['gru_prob']), 3), 'pred': 'Negativo'},
        'fp': {'text': fp_sample['review'][:240], 'target': 'Negativo', 'prob': round(float(fp_sample['gru_prob']), 3), 'pred': 'Positivo'},
        'fn': {'text': fn_sample['review'][:240], 'target': 'Positivo', 'prob': round(float(fn_sample['gru_prob']), 3), 'pred': 'Negativo'}
    }
}

metrics_json_path = os.path.join(OUTPUT_DIR, "results_metrics.json")
with open(metrics_json_path, 'w', encoding='utf-8') as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)

print(f"\n[*] Métricas guardadas exitosamente en {metrics_json_path}")
print("[*] Proceso experimental finalizado con éxito.")

