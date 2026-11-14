#!/usr/bin/env bash
# scanPHY setup for NVIDIA Jetson Orin (JetPack 5.x / 6.x, Ubuntu 20.04 / 22.04).
#
#   ./scripts/setup_jetson.sh              # venv + dependencies
#   ./scripts/setup_jetson.sh --torch      # + PyTorch (CUDA) for on-device training
#   ./scripts/setup_jetson.sh --service    # + start automatically at boot (systemd)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WANT_TORCH=0; WANT_SERVICE=0
for a in "$@"; do
  case "$a" in
    --torch) WANT_TORCH=1 ;;
    --service) WANT_SERVICE=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

b() { printf '\033[1m%s\033[0m\n' "$*"; }
ok() { printf '  \033[1m✓\033[0m %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*"; }

b "scanPHY setup  ($ROOT)"
if [ -f /etc/nv_tegra_release ]; then ok "Jetson: $(head -1 /etc/nv_tegra_release | cut -c1-60)"; else warn "not a Jetson - continuing anyway"; fi

b "1/4  system packages"
if command -v apt-get >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-pip python3-dev libopenblas-dev >/dev/null
  ok "python3-venv, pip, openblas"
fi

b "2/4  python environment (.venv, sees JetPack's system packages)"
if [ ! -d "$ROOT/.venv" ]; then python3 -m venv --system-site-packages "$ROOT/.venv"; fi
"$ROOT/.venv/bin/python" -m pip install -q --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -q -r "$ROOT/requirements.txt" -r "$ROOT/requirements-tools.txt"
ok "numpy, aiohttp, esptool installed"

b "3/4  PyTorch (training)"
if "$ROOT/.venv/bin/python" -c "import torch" 2>/dev/null; then
  ok "torch $("$ROOT/.venv/bin/python" -c 'import torch;print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")')"
elif [ "$WANT_TORCH" = 1 ]; then
  "$ROOT/scripts/install_torch_jetson.sh" "$ROOT/.venv/bin/python"
else
  warn "torch not installed - the dashboard and inference work, training needs: $0 --torch"
fi

b "4/4  network + service"
if command -v ufw >/dev/null && sudo ufw status | grep -q active; then
  sudo ufw allow 5005/udp >/dev/null; sudo ufw allow 3000/tcp >/dev/null; sudo ufw allow 3001/tcp >/dev/null
  ok "firewall: UDP 5005, TCP 3000/3001 open"
fi
mkdir -p "$ROOT/data/recordings" "$ROOT/models"
[ -f "$ROOT/config/room.json" ] || cp "$ROOT/config/room.example.json" "$ROOT/config/room.json"
if [ "$WANT_SERVICE" = 1 ]; then
  sed -e "s#__ROOT__#$ROOT#g" -e "s#__USER__#$USER#g" "$ROOT/scripts/scanphy.service" | sudo tee /etc/systemd/system/scanphy.service >/dev/null
  sudo systemctl daemon-reload && sudo systemctl enable --now scanphy
  ok "systemd service 'scanphy' enabled (journalctl -u scanphy -f)"
fi
if command -v nvpmodel >/dev/null; then warn "tip: sudo nvpmodel -m 0 && sudo jetson_clocks   (max performance for training)"; fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
b "done."
echo "  run:        $ROOT/.venv/bin/python -m scanphy serve"
echo "  demo:       $ROOT/.venv/bin/python -m scanphy serve --source sim"
echo "  dashboard:  http://${IP:-localhost}:3000"
echo "  nodes:      python tools/flash_nodes.py --ssid YOUR_WIFI --target-ip ${IP:-JETSON_IP}"
