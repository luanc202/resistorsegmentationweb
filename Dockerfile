FROM python:3.11.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /server

COPY server/ .

RUN pip install --user pipenv
RUN pipenv install
RUN pipenv shell

EXPOSE 5000

CMD ["python", "app.py"]
