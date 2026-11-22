# PRACTICA: MICROSERVICIO DE INTELIGENCIA ARTIFICIAL
# Dataset: Iris
# Alumno: José Jaime García Mendoza, Mairyn Torres Popocatl, Elizabeth Hernandez Corte
# # Docente: Saúl Olaf Meléndez Loaiza
# Institución: Instituto Tecnológico de Apizaco

# VERSION 2: Entrenamiento del modelo de clasificación

import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train():

    # 1. Cargar el dataset Iris
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Entrenar el algoritmo de clasificación
    clf = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # 4. Evaluar la exactitud del modelo
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(
        f"Modelo entrenado exitosamente. "
        f"Exactitud en prueba: {acc * 100:.2f}%"
    )

    # 5. Exportar el modelo entrenado
    artifact = {
        "model": clf,
        "target_names": iris.target_names.tolist()
    }

    joblib.dump(artifact, "iris_model.pkl")

    print("Artefacto 'iris_model.pkl' guardado correctamente.")


if __name__ == "__main__":
    train()