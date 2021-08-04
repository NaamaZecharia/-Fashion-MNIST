FROM python:3.7.3-stretch
RUN mkdir /app
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python3","app.py"]
