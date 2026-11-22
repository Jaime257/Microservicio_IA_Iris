# PRACTICA: MICROSERVICIO DE INTELIGENCIA ARTIFICIAL
# Dataset: Iris
# Alumno: Jose Jaime Garcia Mendoza, Mairyn Torres Popocatl, Elizabeth Hernandez Corte
# Docente: Saul Olaf Melendez Loaiza
# Instituto Tecnologico de Apizaco

# VERSION 2: Microservicio REST funcional con FastAPI

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# 1. Inicializar la aplicacion FastAPI

app = FastAPI(
    title="Microservicio de Inferencia Iris",
    description="API REST contenerizada para prediccion de especies de la flor Iris.",
    version="1.0.0"
)


# 2. Cargar el modelo entrenado

try:
    artifact = joblib.load("iris_model.pkl")

    model = artifact["model"]
    target_names = artifact["target_names"]

except Exception as e:
    raise RuntimeError(
        f"Error al cargar el archivo del modelo: {e}"
    )


# 3. Definir los datos que recibira la API

class IrisRequest(BaseModel):

    sepal_length: float = Field(
        ...,
        description="Longitud del sepalo en cm",
        examples=[5.1]
    )

    sepal_width: float = Field(
        ...,
        description="Ancho del sepalo en cm",
        examples=[3.5]
    )

    petal_length: float = Field(
        ...,
        description="Longitud del petalo en cm",
        examples=[1.4]
    )

    petal_width: float = Field(
        ...,
        description="Ancho del petalo en cm",
        examples=[0.2]
    )


# 4. Endpoint para comprobar el funcionamiento

@app.get("/")
def read_root():

    return {
        "status": "online",
        "service": "Iris Classifier Microservice",
        "dataset_info": "150 instancias, 4 caracteristicas, 3 clases"
    }


# 5. Endpoint para realizar predicciones

@app.post("/predict")
def predict_species(request: IrisRequest):

    try:

        # Convertir los valores recibidos en un arreglo

        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]])


        # Realizar la prediccion

        prediction_id = int(
            model.predict(features)[0]
        )

        probabilities = model.predict_proba(features)[0]


        # Obtener el nombre de la especie

        predicted_class = target_names[prediction_id]

        confidence = float(
            probabilities[prediction_id]
        )


        # Devolver los resultados en formato JSON

        return {
            "prediction": predicted_class,
            "class_id": prediction_id,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(prob), 4)
                for name, prob in zip(
                    target_names,
                    probabilities
                )
            }
        }

    except Exception as err:

        raise HTTPException(
            status_code=500,
            detail=f"Error en la inferencia: {str(err)}"
        )