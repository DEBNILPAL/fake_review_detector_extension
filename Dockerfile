# Multi-stage build for Node (backend) + Python (deep learning)
# Final image based on Debian with Node LTS and Python 3.11
FROM node:20-bullseye

# Install Python 3.11 and pip
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
 && ln -sf /usr/bin/python3 /usr/local/bin/python \
 && python --version \
 && pip3 --version \
 && rm -rf /var/lib/apt/lists/*

# Create app directories
WORKDIR /app

# Copy dependency manifests first for better layer caching
# Node
COPY backend/package*.json ./backend/
# Python
COPY ["deep learning/requirements.txt", "deep learning/requirements.txt"]

# Install dependencies
RUN cd backend && npm install --omit=dev \
 && cd "/app/deep learning" && pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend ./backend
COPY ["deep learning", "deep learning"]

# Environment
ENV NODE_ENV=production \
    PORT=8000 \
    PYTHON_PATH=/usr/local/bin/python

# Expose port (Render will set $PORT)
EXPOSE 8000

# Start server
WORKDIR /app/backend
CMD ["node", "server.js"]
