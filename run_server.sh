#!/usr/bin/env bash
# ==============================================================================
# run_server.sh — Build Docker image, start container, start Cloudflare Tunnel
#
# Usage:
#   chmod +x run_server.sh
#   ./run_server.sh              # Build image + start everything
#   ./run_server.sh --no-build   # Skip Docker build (use existing image)
# ==============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="banana-detection"
CONTAINER_NAME="banana-detection-api"
HOST_PORT=8000
TUNNEL_NAME="server-nha-lam"   # <-- Change to your cloudflared tunnel name
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/server_startup.log"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()    { echo -e "\n${CYAN}══════════════════════════════════════════${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}══════════════════════════════════════════${NC}"; }
log_file()    { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

# ── Argument Parsing ──────────────────────────────────────────────────────────
NO_BUILD=false
for arg in "$@"; do
  case $arg in
    --no-build) NO_BUILD=true ;;
    *) log_warn "Unknown argument: $arg" ;;
  esac
done

# ── Setup logs ────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
log_file "=== STARTING BANANA DETECTION SERVER ==="

# ── Prerequisites check ───────────────────────────────────────────────────────
log_step "Checking prerequisites"

if ! command -v docker &>/dev/null; then
  log_error "Docker not found. Install Docker Desktop."
  exit 1
fi
log_success "Docker: $(docker --version)"

if ! command -v cloudflared &>/dev/null; then
  log_warn "cloudflared not found — Cloudflare Tunnel will be skipped."
  SKIP_TUNNEL=true
else
  SKIP_TUNNEL=false
  log_success "cloudflared: $(cloudflared --version 2>&1 | head -1)"
fi

# ── Verify model file ─────────────────────────────────────────────────────────
log_step "Verifying model file"
MODEL_FILE="$PROJECT_DIR/models/ultimate_model.pt"
if [ -f "$MODEL_FILE" ]; then
  log_success "Found: models/ultimate_model.pt"
else
  log_error "Missing: models/ultimate_model.pt — place the model file in ./models/ before running."
  exit 1
fi

# # ── Build Docker image ────────────────────────────────────────────────────────
# if [ "$NO_BUILD" = false ]; then
#   log_step "Building Docker image: $IMAGE_NAME"
#   log_file "Building Docker image..."
#   docker build -t "$IMAGE_NAME" "$PROJECT_DIR" 2>&1 | tee -a "$LOG_FILE"
#   log_success "Image built successfully."
#   log_file "Docker image built."
# else
#   log_warn "Skipping build (--no-build flag set)"
# fi

# ── Stop existing container if running ───────────────────────────────────────
if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
  log_warn "Container '$CONTAINER_NAME' already running — stopping it first."
  docker stop "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME"
  log_file "Stopped existing container."
fi

# ── Start Docker container ────────────────────────────────────────────────────
log_step "Starting Docker container"
log_file "Starting container..."

docker run -d \
  --name "$CONTAINER_NAME" \
  -p "${HOST_PORT}:8000" \
  -v "$PROJECT_DIR/models:/app/models" \
  --restart unless-stopped \
  "$IMAGE_NAME"

log_success "Container started: $CONTAINER_NAME"
log_file "Container started."

# ── Wait for API to be healthy ────────────────────────────────────────────────
log_step "Waiting for API to be ready (max 90s)"
MAX_WAIT=90; ELAPSED=0; INTERVAL=5

while [ $ELAPSED -lt $MAX_WAIT ]; do
  if curl -sf "http://localhost:${HOST_PORT}/health" > /dev/null 2>&1; then
    log_success "API is healthy! (${ELAPSED}s elapsed)"
    log_file "API healthy."
    break
  fi
  echo -n "."
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
  log_error "API did not become healthy within ${MAX_WAIT}s."
  log_error "Check logs: docker logs $CONTAINER_NAME"
  log_file "ERROR: API health check timed out."
  exit 1
fi

# ── Start Cloudflare Tunnel ───────────────────────────────────────────────────
if [ "$SKIP_TUNNEL" = false ]; then
  log_step "Starting Cloudflare Tunnel: $TUNNEL_NAME"
  log_file "Starting Cloudflare Tunnel..."
  cloudflared tunnel run "$TUNNEL_NAME" >> "$LOG_DIR/tunnel.log" 2>&1 &
  sleep 5
  log_success "Cloudflare Tunnel started in background."
  log_file "Tunnel started."
else
  log_warn "Cloudflare Tunnel skipped — API accessible locally only."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Banana Bunch Detection — Running         ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  API:      http://localhost:${HOST_PORT}           ║${NC}"
echo -e "${GREEN}║  Health:   http://localhost:${HOST_PORT}/health     ║${NC}"
echo -e "${GREEN}║  Docs:     http://localhost:${HOST_PORT}/docs       ║${NC}"
echo -e "${GREEN}║  Public:   https://rexsantech.com            ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Stop:     ./stop_server.sh                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""

log_file "=== SERVER STARTUP COMPLETE ==="
