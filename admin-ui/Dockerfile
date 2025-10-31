FROM node:18-alpine AS build
WORKDIR /ui

# Accept API base URL at build time (optional; defaults to relative URLs)
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY package.json package-lock.json ./
COPY tsconfig.json vite.config.ts ./
COPY index.html ./index.html
COPY src ./src
COPY tailwind.config.ts ./tailwind.config.ts
COPY postcss.config.js ./postcss.config.js
RUN npm ci || npm i
RUN npm run build

FROM nginx:1.27-alpine
WORKDIR /usr/share/nginx/html

# Copy built assets under /admin/
COPY --from=build /ui/dist /usr/share/nginx/html/admin

# Minimal nginx config to serve /admin without rewriting prefix
RUN mkdir -p /etc/nginx/conf.d
RUN printf '%s\n' \
  'server {' \
  '  listen 80;' \
  '  server_name _;' \
  '  root /usr/share/nginx/html;' \
  '  index index.html;' \
  '  # Handle /admin (no trailing slash) - redirect to /admin/' \
  '  location = /admin {' \
  '    return 301 /admin/;' \
  '  }' \
  '  # Serve Admin UI under /admin/' \
  '  location /admin/ {' \
  '    alias /usr/share/nginx/html/admin/;' \
  '    try_files $uri $uri/ /admin/index.html;' \
  '  }' \
  '}' > /etc/nginx/conf.d/default.conf

EXPOSE 80


