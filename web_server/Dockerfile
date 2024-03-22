FROM continuumio/miniconda3:latest

WORKDIR /app/web_server/

COPY /web_server_data /app/web_server/

RUN conda env create -f /app/web_server/environment.yaml

CMD ["conda", "run", "-n", "web_server", "python", "web_server.py"]
