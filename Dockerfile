FROM python:3.11-slim-buster

RUN apt update -y && apt install awscli -y

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]