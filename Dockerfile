FROM nvidia/cuda:11.2.0-base-ubuntu20.04

RUN apt update
RUN apt-get install -y python3 python3-pip

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir --user -r /requirements.txt

WORKDIR /app
COPY . /app/artifacts

# Set the default command to python3
CMD ["python3"]

EXPOSE 6006 8080