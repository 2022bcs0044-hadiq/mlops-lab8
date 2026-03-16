FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install dvc[s3] pandas scikit-learn joblib

CMD ["python","src/train.py"]