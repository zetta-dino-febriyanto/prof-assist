FROM python:3.10

WORKDIR /prof-assist

COPY . .

RUN apt-get update

RUN apt-get install -y python3-pip

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "app.py"]