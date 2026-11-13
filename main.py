import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Импорт классификаторов
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    """Загрузка и подготовка данных"""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Ошибка: Файл {filepath} не найден. Убедитесь, что он лежит в той же папке.")
        return None, None, None, None

    # Удаляем столбец Id, если он есть
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    # Разделяем признаки (X) и целевую переменную (y)
    X = df.drop('Species', axis=1)
    y = df['Species']

    # Кодируем целевую переменную
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Разделяем на обучающую и тестовую выборки (70% / 30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, scaler_type="standard"):
    
    if scaler_type == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_used = scaler.fit_transform(X_train)
        X_test_used = scaler.transform(X_test)

    elif scaler_type == "binarize":
        thresholds = np.median(X_train, axis=0)
        X_train_used = (X_train > thresholds).astype(int)
        X_test_used = (X_test > thresholds).astype(int)

    else:
        X_train_used = X_train
        X_test_used = X_test
    
    model.fit(X_train_used, y_train)

    y_pred = model.predict(X_test_used)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_used)
        auc_roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_used)
        auc_roc = roc_auc_score(y_test, y_score, multi_class='ovr', average='weighted')
    else:
        auc_roc = np.nan

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc
    }


def main():
    filepath = 'Iris.csv'
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    
    if X_train is None:
        return

    models = {
        "GaussianNB":      (GaussianNB(),                            "standard"),
        "MultinomialNB":   (MultinomialNB(),                         "minmax"),
        "ComplementNB":    (ComplementNB(),                          "minmax"),
        "BernoulliNB":     (BernoulliNB(),                           "binarize"),
        
        "Decision Tree":   (DecisionTreeClassifier(random_state=42), "standard"),
        
        "LDA":             (LinearDiscriminantAnalysis(),            "standard"),
        
        "SVM":             (SVC(probability=True, random_state=42),  "standard"),
        
        "kNN":             (KNeighborsClassifier(n_neighbors=5),     "standard"),
    }

    results = []

    print(f"{'='*20} Результаты классификации {'='*20}\n")
    
    for name, (model, scaler_type) in models.items():
        try:
            res = evaluate_model(model, X_train, X_test, y_train, y_test, name, scaler_type)
            results.append(res)
        except Exception as e:
            print(f"Ошибка при обучении модели {name}: {e}")

    results_df = pd.DataFrame(results)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()