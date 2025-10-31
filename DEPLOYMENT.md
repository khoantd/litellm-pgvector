# Deployment Guide: Separated Frontend and Backend

This guide explains how to deploy the Vector Store API with separated frontend and backend services.

## Architecture

- **Backend API**: FastAPI application (port 8003 in container)
- **Frontend UI**: Nginx serving static assets (port 3006 in container)

## Option 1: Same Domain with Traefik (Recommended)

Both services behind the same domain, Traefik routes by path.

### Build Images

```bash
# Backend (defaults to linux/amd64)
./scripts/build-backend.sh

# Backend for specific platform (e.g., ARM64)
PLATFORM=linux/arm64 ./scripts/build-backend.sh

# Frontend (no VITE_API_BASE_URL needed - uses relative URLs, defaults to linux/amd64)
./scripts/build-frontend.sh

# Frontend for specific platform
PLATFORM=linux/arm64 ./scripts/build-frontend.sh

# Build and push with platform
PLATFORM=linux/amd64 PUSH=true ./scripts/build-backend.sh
PLATFORM=linux/amd64 PUSH=true ./scripts/build-frontend.sh
```

### Traefik Labels (Docker Compose Example)

```yaml
version: '3.8'

services:
  backend:
    image: vector-store-api:latest
    labels:
      - "traefik.http.routers.vsapi.rule=Host(`your-domain.com`) && (PathPrefix(`/v1`) || Path(`/health`))"
      - "traefik.http.routers.vsapi.entrypoints=websecure"
      - "traefik.http.routers.vsapi.tls.certresolver=letsencrypt"
      - "traefik.http.services.vsapi.loadbalancer.server.port=8003"
    env_file:
      - .env
    networks:
      - traefik

  frontend:
    image: vector-store-admin:latest
    labels:
      - "traefik.http.routers.vsui.rule=Host(`your-domain.com`) && PathPrefix(`/admin`)"
      - "traefik.http.routers.vsui.entrypoints=websecure"
      - "traefik.http.routers.vsui.tls.certresolver=letsencrypt"
      - "traefik.http.services.vsui.loadbalancer.server.port=3006"
    networks:
      - traefik

networks:
  traefik:
    external: true
```

**Important**: Do NOT use `StripPrefix` middleware - the `/admin` prefix must be preserved.

### How It Works

- User visits: `https://your-domain.com/admin`
- Frontend makes calls to `/v1/admin/settings` (relative URL)
- Traefik routes `/v1/*` to backend service
- Traefik routes `/admin/*` to frontend service
- Same origin = no CORS issues

## Option 2: Different Domains/Subdomains

Backend and frontend on separate domains/subdomains.

### Build Frontend with API Base URL

```bash
# With platform option
VITE_API_BASE_URL="https://api.your-domain.com" \
  PLATFORM=linux/amd64 \
  ./scripts/build-frontend.sh
```

Or in Dockerfile directly:

```bash
docker build \
  --build-arg VITE_API_BASE_URL="https://api.your-domain.com" \
  -t vector-store-admin:latest \
  -f admin-ui/Dockerfile \
  admin-ui/
```

### Traefik Labels

```yaml
services:
  backend:
    image: vector-store-api:latest
    labels:
      - "traefik.http.routers.vsapi.rule=Host(`api.your-domain.com`)"
      - "traefik.http.routers.vsapi.entrypoints=websecure"
      - "traefik.http.routers.vsapi.tls.certresolver=letsencrypt"
      - "traefik.http.services.vsapi.loadbalancer.server.port=8003"
    env_file:
      - .env

  frontend:
    image: vector-store-admin:latest
    labels:
      - "traefik.http.routers.vsui.rule=Host(`admin.your-domain.com`)"
      - "traefik.http.routers.vsui.entrypoints=websecure"
      - "traefik.http.routers.vsui.tls.certresolver=letsencrypt"
      - "traefik.http.services.vsui.loadbalancer.server.port=3006"
```

### CORS Configuration

Backend already has CORS enabled (`allow_origins=["*"]` in `main.py`). If you want to restrict:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://admin.your-domain.com"],  # Frontend domain only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Option 3: Docker Compose (No Traefik)

Direct port mapping for testing/development.

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8003:8003"
    env_file:
      - .env

  frontend:
    build:
      context: .
      dockerfile: admin-ui/Dockerfile
      args:
        VITE_API_BASE_URL: "http://localhost:8003"  # or backend service name
    ports:
      - "3006:3006"
```

**Note**: For CORS to work, `VITE_API_BASE_URL` must match the exact URL users will access (including protocol).

## Verification

1. **Backend Health**: `curl https://your-domain.com/health`
2. **Frontend Assets**: `curl https://your-domain.com/admin/assets/index-*.css` (should return CSS)
3. **API from Frontend**: Open browser console on `/admin`, verify network requests succeed

## Database Migrations

The `app_settings` table (and other tables) must be created in your database before the API can function.

### Option 1: Automatic Migration on Startup (Recommended)

Add to your backend `.env`:
```bash
AUTO_MIGRATE=true
```

The backend will run `prisma db push` automatically on startup. This is **incremental and safe**:
- **Creates missing tables** if they don't exist
- **Adds new columns** to existing tables without dropping data
- **Does NOT drop** existing tables or columns
- **Skips changes** if schema is already up to date (fast)

You can safely keep `AUTO_MIGRATE=true` in production - it will only apply new schema changes when they exist.

### Option 2: Manual Migration Script

Run migrations manually:
```bash
# Set DATABASE_URL in your environment
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"

# Run migration script
./scripts/migrate.sh
```

Or directly with Prisma:
```bash
prisma generate
prisma db push --accept-data-loss
```

### Option 3: Run Migration in Container

```bash
# Run migration inside the running container
docker exec -it <backend-container> prisma db push --accept-data-loss

# Or create a one-off container for migration
docker run --rm --env-file .env <backend-image> prisma db push --accept-data-loss
```

### Migration Error: "relation app_settings does not exist"

This means the database tables haven't been created. Run one of the migration options above.

## Troubleshooting

### CSS/JS Not Loading

- Check Traefik routes preserve `/admin` prefix
- Verify `base: '/admin/'` in `vite.config.ts`
- Check browser console for 404s on asset URLs

### CORS Errors (Different Domains)

- Verify `VITE_API_BASE_URL` matches backend domain exactly
- Check backend CORS `allow_origins` includes frontend domain
- Ensure backend serves on HTTPS (required for secure origins)

### API Calls Return 401

- Verify `SERVER_API_KEY` in backend `.env`
- Check Authorization header format: `Bearer <key>`
- Test with curl: `curl -H "Authorization: Bearer $KEY" https://api.domain.com/v1/admin/settings`

### Database Migration Errors

- Ensure `DATABASE_URL` is correct and accessible
- Check PostgreSQL extensions: `CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; CREATE EXTENSION IF NOT EXISTS vector;`
- Verify database user has CREATE TABLE permissions
- Check logs for specific Prisma errors

