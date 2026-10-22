FROM python:3.11-slim

WORKDIR /app

COPY . /app
COPY ../../data /app/data

# Install system dependencies (FIXES lightgbm error)

RUN apt-get update && apt-get install -y libgomp1
# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn \
    xgboost lightgbm \
    imbalanced-learn \
    fastapi uvicorn \
    kafka-python-ng \
    joblib shap \
    SQLAlchemy psycopg2-binary \
    requests pyyaml

CMD ["python", "app.py"]