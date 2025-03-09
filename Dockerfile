FROM python:3.11.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /server

COPY server/ .

RUN pip install --no-cache-dir -r requirements.txt

RUN openssl req -x509 -newkey rsa:2048 -keyout 192.168.15.14-key.pem -out 192.168.15.14.pem -days 365 -nodes \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=192.168.15.14"

EXPOSE 5000

CMD ["python", "server.py"]
