FROM continuumio/miniconda3:latest

WORKDIR /app/ml_server/

COPY /ml_server_data /app/ml_server/

RUN conda env create -f /app/ml_server/environment.yaml

CMD ["conda", "run", "-n", "ml_server", "python", "ml_server.py"]
