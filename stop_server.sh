#!/usr/bin/env bash
# ==============================================================================
# stop_server.sh — Stop Docker container and Cloudflare Tunnel
# ==============================================================================

CONTAINER_NAME="banana-detection-api"
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
LOG_FILE="$LOG_DIR/server_stop.log"

mkdir -p "$LOG_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- STOPPING BANANA DETECTION SERVER ---" >> "$LOG_FILE"

echo "Stopping Docker container: $CONTAINER_NAME..."
if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
  docker stop "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Container stopped." >> "$LOG_FILE"
  echo "Container stopped."
else
  echo "Container '$CONTAINER_NAME' is not running."
fi

echo "Stopping Cloudflare Tunnel..."
# Windows (Git Bash)
if command -v taskkill &>/dev/null; then
  taskkill //F //IM cloudflared.exe >> "$LOG_FILE" 2>&1 || true
else
  # Linux / macOS
  pkill -f "cloudflared tunnel" >> "$LOG_FILE" 2>&1 || true
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Tunnel stopped." >> "$LOG_FILE"
echo "Cloudflare Tunnel stopped."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- ALL SERVICES STOPPED ---" >> "$LOG_FILE"
echo "Done!"
