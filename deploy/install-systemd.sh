#!/usr/bin/env bash
# install-systemd.sh — Install airML as a systemd service.
#
# Usage (as root):
#   sudo bash install-systemd.sh
#
# What this script does:
#   1. Creates the `airml` system user and group.
#   2. Copies airml.service to /etc/systemd/system/.
#   3. Reloads systemd, enables and starts the service.
#
# Prerequisites:
#   - airml binary at /usr/local/bin/airml
#   - libonnxruntime.so at /usr/local/lib/libonnxruntime.so
#     (run `airml install-runtime` first, or install manually)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="${SCRIPT_DIR}/airml.service"
SYSTEMD_DIR="/etc/systemd/system"
AIRML_USER="airml"
AIRML_GROUP="airml"
AIRML_BIN="/usr/local/bin/airml"
ORT_LIB="/usr/local/lib/libonnxruntime.so"

# ─── 1. Root check ────────────────────────────────────────────────────────────
if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: This script must be run as root (e.g. sudo bash install-systemd.sh)" >&2
    exit 1
fi

# ─── 2. Preflight checks ─────────────────────────────────────────────────────
if [[ ! -f "${AIRML_BIN}" ]]; then
    echo "ERROR: airml binary not found at ${AIRML_BIN}" >&2
    echo "       Download from https://github.com/rlaope/airML/releases or build from source." >&2
    exit 1
fi

if [[ ! -f "${ORT_LIB}" ]]; then
    echo "WARNING: ONNX Runtime not found at ${ORT_LIB}" >&2
    echo "         Run: airml install-runtime   (or install ORT manually)" >&2
    echo "         Continuing — the service will fail to start until ORT is installed." >&2
fi

if [[ ! -f "${SERVICE_FILE}" ]]; then
    echo "ERROR: Service file not found at ${SERVICE_FILE}" >&2
    exit 1
fi

# ─── 3. Create system user and group ─────────────────────────────────────────
if ! getent group "${AIRML_GROUP}" > /dev/null 2>&1; then
    echo "Creating group: ${AIRML_GROUP}"
    groupadd --system "${AIRML_GROUP}"
fi

if ! getent passwd "${AIRML_USER}" > /dev/null 2>&1; then
    echo "Creating user: ${AIRML_USER}"
    useradd \
        --system \
        --gid "${AIRML_GROUP}" \
        --no-create-home \
        --shell /usr/sbin/nologin \
        --comment "airML inference server" \
        "${AIRML_USER}"
fi

# ─── 4. Create state directory ───────────────────────────────────────────────
mkdir -p /var/lib/airml
chown "${AIRML_USER}:${AIRML_GROUP}" /var/lib/airml
chmod 750 /var/lib/airml

# ─── 5. Install service file ─────────────────────────────────────────────────
echo "Installing ${SERVICE_FILE} -> ${SYSTEMD_DIR}/airml.service"
cp "${SERVICE_FILE}" "${SYSTEMD_DIR}/airml.service"
chmod 644 "${SYSTEMD_DIR}/airml.service"

# ─── 6. Reload, enable, start ────────────────────────────────────────────────
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling airml.service (start on boot)..."
systemctl enable airml.service

echo "Starting airml.service..."
systemctl start airml.service

# ─── 7. Status ───────────────────────────────────────────────────────────────
echo ""
echo "airML service installed successfully."
echo ""
systemctl status airml.service --no-pager
