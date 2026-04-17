"""
visualization.py
Todas las visualizaciones del proyecto: WordCloud, N-gramas,
frecuencia de palabras, matrices de confusión y curvas de entrenamiento.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix


# =============================================================================
#  WORDCLOUDS
# =============================================================================

def plot_wordcloud(text: str, title: str = 'WordCloud',
                   width: int = 800, height: int = 400,
                   max_words: int = 200) -> None:
    """Genera y muestra una nube de palabras a partir de un string."""
    wc = WordCloud(
        width=width, height=height,
        background_color='white',
        max_words=max_words,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_wordclouds_by_sentiment(df: pd.DataFrame,
                                  text_col: str = 'review_lematizer_str',
                                  sentiment_col: str = 'sentiment_map') -> None:
    """
    Muestra tres WordClouds: general, positivo y negativo.
    """
    text_all = ' '.join(df[text_col])
    text_pos = ' '.join(df[df[sentiment_col] == 1][text_col])
    text_neg = ' '.join(df[df[sentiment_col] == 0][text_col])

    plot_wordcloud(text_all, title='WordCloud - General')
    plot_wordcloud(text_pos, title='WordCloud - Reviews Positivas')
    plot_wordcloud(text_neg, title='WordCloud - Reviews Negativas')


# =============================================================================
#  N-GRAMAS
# =============================================================================

def plot_ngram_wordcloud(df: pd.DataFrame,
                          text_col: str = 'review_lematizer_str',
                          sentiment_col: str = 'sentiment_map',
                          sentiment_value: int | None = None,
                          ngram_range: tuple = (3, 3),
                          stop_words: list | None = None,
                          title: str = 'N-Gramas WordCloud') -> None:
    """
    Genera un WordCloud de N-gramas.
    sentiment_value: 1=positivo, 0=negativo, None=general.
    """
    if sentiment_value is not None:
        subset = df[df[sentiment_col] == sentiment_value]
    else:
        subset = df

    vec = CountVectorizer(ngram_range=ngram_range,
                          stop_words=stop_words or 'english')
    X   = vec.fit_transform(subset[text_col])

    freq = dict(zip(vec.get_feature_names_out(), X.toarray().sum(axis=0)))

    wc = WordCloud(width=800, height=400, background_color='white') \
         .generate_from_frequencies(freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
#  FRECUENCIA DE PALABRAS (BARRAS)
# =============================================================================

def plot_top_words(df: pd.DataFrame,
                   token_col: str = 'review_lematizer',
                   sentiment_col: str = 'sentiment_map',
                   top_n: int = 10) -> None:
    """
    Muestra tres gráficos de barras: general, positivo y negativo.
    token_col debe contener listas de tokens.
    """
    configs = [
        (None,  'skyblue', 'Top palabras - General'),
        (1,     'green',   'Top palabras - Positivas'),
        (0,     'red',     'Top palabras - Negativas'),
    ]

    for val, color, title in configs:
        subset = df if val is None else df[df[sentiment_col] == val]
        tokens = [t for lst in subset[token_col] for t in lst]
        words, freqs = zip(*Counter(tokens).most_common(top_n))

        plt.figure(figsize=(10, 6))
        plt.bar(words, freqs, color=color)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel('Palabras lematizadas')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# =============================================================================
#  CURVAS DE ENTRENAMIENTO (RNN / LSTM / GRU)
# =============================================================================

def plot_history(history, model_name: str = 'Modelo') -> None:
    """
    Plotea las curvas de accuracy y loss de un historial de entrenamiento Keras.
    """
    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(history.history['accuracy'],     label='Train',     linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validación', linewidth=2, linestyle='--')
    axes[0].set_title('Curva de Precisión', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'],     label='Train',     linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validación', linewidth=2, linestyle='--')
    axes[1].set_title('Curva de Pérdida', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Épocas')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.suptitle(f'Rendimiento del {model_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
#  MATRICES DE CONFUSIÓN
# =============================================================================

def plot_confusion_matrices(models_results: list[tuple]) -> None:
    """
    Muestra matrices de confusión para múltiples modelos en una sola figura.

    Parámetros
    ----------
    models_results : lista de tuplas (nombre, y_true, y_pred)
    """
    n = len(models_results)
    sns.set_theme(style='white')
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))

    if n == 1:
        axes = [axes]

    for ax, (nombre, y_true, y_pred) in zip(axes, models_results):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negativo', 'Positivo'],
                    yticklabels=['Negativo', 'Positivo'],
                    ax=ax)
        ax.set_title(f'Matriz de Confusión\n{nombre}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')

    plt.suptitle('Comparación de Modelos', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
#  TABLA RESUMEN DE MÉTRICAS
# =============================================================================

def metrics_table(models_results: list[tuple]) -> pd.DataFrame:
    """
    Construye una tabla comparativa de métricas.

    Parámetros
    ----------
    models_results : lista de tuplas (nombre, y_true, y_pred)

    Retorna
    -------
    DataFrame ordenado por F1-Score descendente.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    rows = []
    for nombre, y_true, y_pred in models_results:
        rows.append({
            'Modelo':    nombre,
            'Accuracy':  round(accuracy_score(y_true, y_pred),  4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'Recall':    round(recall_score(y_true, y_pred),    4),
            'F1-Score':  round(f1_score(y_true, y_pred),        4),
        })

    return pd.DataFrame(rows).sort_values('F1-Score', ascending=False)
