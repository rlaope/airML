# systemd

The recommended deployment method for bare-metal Linux servers (Debian, Ubuntu, RHEL, and derivatives).

## 1. Install the binary

```bash
curl -L https://github.com/rlaope/airML/releases/latest/download/airml-linux-x86_64.tar.gz \
  | tar xz
sudo install -m 755 airml /usr/local/bin/airml
```

For ARM64 (Graviton/Ampere), use `airml-linux-aarch64.tar.gz`.

## 2. Install ONNX Runtime

```bash
sudo airml install-runtime
```

Note the printed `ORT_DYLIB_PATH` — you will need it in the service file.

## 3. Run the installer script

```bash
sudo bash deploy/install-systemd.sh
```

This script:
- Creates a `airml` system user
- Installs `deploy/airml.service` to `/etc/systemd/system/`
- Enables and starts the service

## 4. Verify

```bash
systemctl status airml
curl http://127.0.0.1:8080/healthz
```

## Service file reference

```ini
[Unit]
Description=airML inference server
After=network.target

[Service]
User=airml
ExecStart=/usr/local/bin/airml serve --bind 127.0.0.1:8080 --log-format json
Environment=ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
Environment=AIRML_CACHE_DIR=/var/cache/airml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

The full unit file is at `deploy/airml.service`.

## Uninstall

```bash
sudo systemctl stop airml
sudo systemctl disable airml
sudo rm /etc/systemd/system/airml.service
sudo systemctl daemon-reload
sudo userdel airml
```

## See also

- [Deployment overview](overview.md)
- [Observability](../operations/observability.md) — logging and metrics
