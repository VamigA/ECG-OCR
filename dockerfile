FROM python:3.12-slim

WORKDIR /ecg-ocr
COPY . .

RUN pip install -U pip && \
	pip install -r requirements.txt