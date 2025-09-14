#!/bin/sh

# Wait for backend service if it exists
if [ "$WAIT_FOR_BACKEND" = "true" ]; then
  echo "Waiting for backend service..."
  while ! wget --quiet --tries=1 --timeout=3 --spider http://backend:8000/health 2>/dev/null; do
    echo "Backend not ready, waiting..."
    sleep 2
  done
  echo "Backend is ready!"
fi

# Ensure session directory exists with correct permissions
mkdir -p /app/whatsapp_session
chmod 755 /app/whatsapp_session

# Start PM2 in runtime mode
exec pm2-runtime start ecosystem.config.js