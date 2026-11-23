
# Imagen base de Python
FROM python:3.12-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar bibliotecas
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos del microservicio
COPY train_model.py app.py ./

# Entrenar y guardar el modelo Iris
RUN python train_model.py

# Puerto del microservicio
EXPOSE 8000

# Iniciar FastAPI con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
