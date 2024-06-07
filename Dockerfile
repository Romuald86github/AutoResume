# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install Nginx
RUN apt-get update && apt-get install -y nginx

# Configure Nginx
COPY nginx.conf /etc/nginx/nginx.conf
COPY src/app/templates /var/www/html

CMD ["/bin/bash", "-c", "nginx -g 'daemon off;' & gunicorn -w 4 -b 0.0.0.0:5000 src.app.app:app"]