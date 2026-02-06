# SSL Setup with Let's Encrypt

## Prerequisites

- A domain name pointing to your server
- Docker and Docker Compose installed
- Port 80 and 443 open on your firewall

## Option 1: Certbot Standalone (Recommended)

### Install Certbot

```bash
sudo apt update
sudo apt install certbot
```

### Obtain Certificate

```bash
# Stop any service on port 80 first
docker compose -f docker-compose.prod.yml down

# Get certificate
sudo certbot certonly --standalone -d your-domain.com

# Certificate files will be at:
# /etc/letsencrypt/live/your-domain.com/fullchain.pem
# /etc/letsencrypt/live/your-domain.com/privkey.pem
```

### Add SSL to docker-compose.prod.yml

Add to the `nginx` service:

```yaml
nginx:
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - /etc/letsencrypt:/etc/letsencrypt:ro
```

### Update nginx.conf for SSL

Add an SSL server block to `deploy/nginx/nginx.conf`:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # ... (copy all location blocks from the port 80 server block)
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}
```

### Auto-Renewal

```bash
# Test renewal
sudo certbot renew --dry-run

# Certbot auto-renews via systemd timer
sudo systemctl status certbot.timer
```

## Option 2: Certbot Docker Container

Use the `certbot/certbot` Docker image if you prefer not to install certbot on the host:

```bash
docker run -it --rm \
  -v /etc/letsencrypt:/etc/letsencrypt \
  -v /var/lib/letsencrypt:/var/lib/letsencrypt \
  -p 80:80 \
  certbot/certbot certonly --standalone -d your-domain.com
```
