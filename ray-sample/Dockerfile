FROM python:3.7.6-slim
WORKDIR /app

RUN apt-get -y update \
  && apt-get -y install -y libglib2.0-0

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY main.py .

ENTRYPOINT ["python", "main.py"]
