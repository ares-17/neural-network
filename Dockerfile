FROM python:3.8

RUN pip install --upgrade pip
RUN pip install opencv-python-headless pandas
RUN pip install matplotlib keras tensorflow

WORKDIR /app
CMD [ "python3", "main.py" ]