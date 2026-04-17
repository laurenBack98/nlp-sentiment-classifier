"""
models/classical.py
Modelos clásicos de ML: Logistic Regression y SVC.
Cada función devuelve el mejor estimador entrenado.
"""

import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# =============================================================================
#  LOGISTIC REGRESSION
# =============================================================================

def train_logistic_regression(
    features: pd.DataFrame,
    text_col: str = 'token',
    target_col: str = 'sentiment_map',
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: int = 0,
) -> tuple:
    """
    Entrena un pipeline TF-IDF + Logistic Regression con HalvingGridSearchCV.

    Retorna
    -------
    model       : mejor estimador entrenado
    x_test      : features de test
    y_test      : etiquetas de test
    y_pred      : predicciones sobre test
    """
    x = features[text_col]
    y = features[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression()),
    ])

    hiperparams = {
        'vectorizer__max_features': [1000, 2000, 5000, 10000],
        'vectorizer__ngram_range':  [(1, 1), (1, 2)],
        'vectorizer__sublinear_tf': [True],
        'classifier__class_weight': ['balanced', None],
        'classifier__penalty':      ['l1', 'l2'],
        'classifier__solver':       ['liblinear'],
        'classifier__C':            [0.01, 0.1, 1, 10, 100],
    }

    search = HalvingGridSearchCV(
        pipeline, param_grid=hiperparams,
        cv=5, n_jobs=-1, verbose=verbose
    )
    search.fit(x_train, y_train)

    model  = search.best_estimator_
    y_pred = model.predict(x_test)

    print(f"[Logistic Regression] Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(f"Mejores parámetros: {search.best_params_}")

    return model, x_test, y_test, y_pred


# =============================================================================
#  SVC (Support Vector Machine)
# =============================================================================

def train_svc(
    features: pd.DataFrame,
    text_col: str = 'token',
    target_col: str = 'sentiment_map',
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: int = 0,
) -> tuple:
    """
    Entrena un pipeline TF-IDF + SVC con HalvingGridSearchCV.

    Retorna
    -------
    model       : mejor estimador entrenado
    x_test      : features de test
    y_test      : etiquetas de test
    y_pred      : predicciones sobre test
    """
    x = features[text_col]
    y = features[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('model_svm',  SVC()),
    ])

    hiperparams = {
        'vectorizer__max_features': [1000, 3000, 5000],
        'vectorizer__ngram_range':  [(1, 1), (1, 2)],
        'vectorizer__sublinear_tf': [True, False],
        'model_svm__C':             [0.1, 1, 10],
        'model_svm__kernel':        ['linear', 'rbf'],
    }

    search = HalvingGridSearchCV(
        pipeline, param_grid=hiperparams,
        scoring='f1', cv=5, n_jobs=-1, verbose=verbose
    )
    search.fit(x_train, y_train)

    model  = search.best_estimator_
    model.fit(x_train, y_train)   # re-entrenamos el mejor sobre todos los datos de train
    y_pred = model.predict(x_test)

    print(f"[SVC] Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(f"Mejores parámetros: {search.best_params_}")

    return model, x_test, y_test, y_pred
