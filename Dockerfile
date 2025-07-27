# FROM python:3.11-slim

# RUN apt update -y

# WORKDIR /app

# COPY . /app
# RUN pip install -r requirements.txt

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Optional: apt update if you need more system packages
RUN apt update -y

WORKDIR /app

# copy requirements.txt 
# COPY requirements.txt /app
COPY . /app

# Install requirements (torch already installed in base image, so remove it from requirements.txt)
RUN pip install -r requirements.txt

# Copy the rest of the application code
# COPY . /app


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
