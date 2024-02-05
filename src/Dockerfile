
FROM python:3.9

WORKDIR /app
ENV PYTHONUNBUFFERED 1
# Copy the source code into the container.
COPY . .

RUN pip install wheel setuptools pip --upgrade
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD uvicorn 'api:app' --host=0.0.0.0 --port=8000
