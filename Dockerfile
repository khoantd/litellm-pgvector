FROM node:18-alpine AS admin_builder
WORKDIR /ui
COPY admin-ui/package.json admin-ui/tsconfig.json admin-ui/vite.config.ts ./
COPY admin-ui/index.html ./index.html
COPY admin-ui/src ./src
RUN npm ci || npm i && npm run build

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Copy built admin UI
COPY --from=admin_builder /ui/dist /app/static/admin

# Generate Prisma client
RUN prisma generate

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"] 