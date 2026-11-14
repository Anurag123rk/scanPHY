# scanPHY

**WiFi CSI human sensing and indoor localisation using 3× ESP32-S3 nodes and an NVIDIA Jetson Orin.**
There's no camera and nothing to wear: the ESP32-S3 boards measure how a human body disturbs the WiFi channel, and
a neural network on the Jetson turns that into presence, activity, position and breathing rate. The results are
shown live in a monochrome 3D dashboard.

scanPHY is a self-contained replacement for a RuView setup. It uses the same ports (UDP 5005 / HTTP 3000 / WS 3001)
and the same 3-board triangle layout, and it adds a model that **trains itself on your room**.

```
 ┌──────────┐  WiFi   ┌────────────┐   CSI over UDP :5005   ┌──────────────────────────────┐
 │  Router  │ ──────▶ │ ESP32-S3 ×3│ ─────────────────────▶ │ Jetson Orin · scanphy serve  │
 └──────────┘ frames  │ scanPHY fw │                        │  DSP → features → NN model   │
      ▲               └────────────┘                        │  tracker · vitals · trainer  │
      └──── pings (≈50 Hz) keep the router transmitting     └──────────────┬───────────────┘
                                                            HTTP :3000 / WS :3001
                                                                            ▼
                                                                  3D dashboard (browser)
```

## Status

| Feature | Status | Notes |
|---|---|---|
| Presence detection | ✅ works | 1+ node |
| Activity (empty / still / moving) | ✅ works | model, ~94 % on simulated rooms |
| Breathing rate | ✅ works | single still person, < 3 m from a link |
| Zone-level position (3×3 grid) | ✅ works after calibration | 3 nodes, triangle placement |
| x/y coordinates | ⚠️ approximate | ~0.7 m error in simulation; calibrate your room |
| Heart rate | 🧪 experimental | shown with an `EXP` tag |
| Multi-person | ❌ not supported | the tracker assumes one person |

The accuracy figures come from the built-in physics simulator. Real rooms vary, so run the calibration wizard (below).

## Repository layout

```
scanphy/            Python package (runs on the Jetson)
  server.py           UDP receiver + REST/WebSocket server + dashboard
  engine.py           live pipeline: buffers → features → model → tracker → vitals
  features.py         feature extraction (shared by inference and training)
  model.py            NumPy-only inference runtime (no PyTorch needed to run)
  trainer.py          PyTorch trainer (CUDA on Jetson), ONNX export
  autotrain.py        background training + hot model swap
  simulator.py        physics-based multipath CSI simulator (training + demo)
  dataset.py          labelled recordings captured from the dashboard
  web/                dashboard (vanilla JS + three.js, works offline)
firmware/scanphy-node/  ESP-IDF firmware for the ESP32-S3 nodes
tools/              provision.py / flash_nodes.py (flash + configure boards)
models/             scanphy-base.npz (pretrained) + .onnx
scripts/            Jetson setup, PyTorch installer, systemd unit
```

## Quick start

### 1. Jetson Orin

```bash
git clone https://github.com/Anurag123rk/scanPHY.git
cd scanPHY
./scripts/setup_jetson.sh --torch        # venv, deps, CUDA PyTorch; add --service to autostart
.venv/bin/python -m scanphy serve --source sim   # demo, no hardware needed
```

Open `http://<jetson-ip>:3000`. In demo mode a simulated person walks around the room. The ◇ marker shows where the
person really is, so you can compare it with the estimate.

### 2. Firmware

Get the binaries in one of two ways:
- **CI:** download the `scanphy-node-firmware` artifact from the GitHub Actions run and unzip it into `firmware/release/`.
- **Build it yourself** with ESP-IDF 5.2+, or with Docker:

```bash
cd firmware/scanphy-node
docker run --rm -v "$PWD:/project" -w /project espressif/idf:v5.2.2 \
  bash -c "idf.py set-target esp32s3 && idf.py build"
```

### 3. Flash and provision all three boards

Plug the boards in over USB, then run:

```bash
pip install -r requirements-tools.txt
python tools/flash_nodes.py --ssid "YOUR_WIFI" --password "YOUR_PASS" --target-ip <jetson-ip>
```

Each board gets node id 1, 2, 3 in turn, and the script then checks that every node is streaming. To configure a
single board:
`python tools/provision.py --port /dev/ttyACM0 --node-id 2 --ssid ... --password ... --target-ip <jetson-ip> --flash-firmware`

Boards still running RuView firmware keep working, because the receiver understands their frame format too.

### 4. Place the nodes and set up the room

- Put the boards in a triangle around the room at desk height (about 1 m). Don't put them all on one wall.
- In **Setup**, enter the room size and the position of the router and of each node, in metres, measured from one
  floor corner.
- Optional: upload a Blender `.glb` of your room. Set units to metres and the origin to the same floor corner.

### 5. Run it

```bash
.venv/bin/python -m scanphy serve          # or: sudo systemctl start scanphy
```

### 6. Calibrate and auto-train (recommended)

Open **Calibrate**:
1. **Capture baseline.** Leave the room for 20 s.
2. **Start wizard.** Stand still at each highlighted spot (10 s each). Then walk around, then leave.
3. When the wizard finishes, the model is **fine-tuned on the Jetson GPU** with your recordings. It is evaluated and
   **hot-swapped** into the live pipeline. Progress and learning curves are on the **Train** tab.

With auto-train enabled, training also starts on its own after 8 new recordings.

## Train anywhere, run on the Jetson

A model is a single `.npz` file. Inference uses NumPy only, so you can train on any machine with PyTorch and load the
result on the Jetson:

```bash
pip install -r requirements.txt -r requirements-train.txt
# generic base model (synthetic data):
python -m scanphy train --synthetic 150000 --config config/room.example.json --epochs 60 --onnx --out models/my-base.npz
# fine-tune on recordings copied from the Jetson (data/recordings/):
python -m scanphy train --recordings data/recordings --config config/room.json \
    --init models/scanphy-base.npz --epochs 60 --onnx --out models/my-room.npz
```

Then **Models → Upload .npz** in the dashboard, or copy the file into `models/` on the Jetson and click **Activate**.
The `.onnx` export has normalisation and softmax built in, so it is ready for TensorRT
(`trtexec --onnx=models/my-room.onnx`).

## Interfaces

| | |
|---|---|
| UDP 5005 | CSI frames, scanPHY v1 format (see `scanphy/protocol.py`); legacy `0xC5110001` frames are also accepted |
| `http://host:3000` | dashboard |
| `ws://host:3000/ws`, `ws://host:3001` | JSON state at 10 Hz: `presence, activity, x, y, confidence, zone, vitals, nodes…` |
| REST | `/api/status`, `/api/config`, `/api/baseline`, `/api/record`, `/api/recordings`, `/api/train`, `/api/models` … |

CLI: `python -m scanphy serve | train | emulate | probe | info`. Use `emulate` to send simulated node traffic over
UDP, and `probe` to list which boards are streaming.

## How it works

1. **Firmware.** Each node pings the router at about 50 Hz and captures CSI (64 LLTF subcarriers) from every frame the
   router sends back. It streams that CSI to the Jetson over UDP.
2. **Features.** For each 1 s window and each node, scanPHY computes a gain-invariant log-amplitude profile across
   13 subcarrier bands. It measures how far that profile deviates from an empty-room baseline and how much it varies
   over time, and adds RSSI and the node/router geometry. That gives 98 values in total.
3. **Model.** A 3-layer MLP with LayerNorm has three output heads: activity (3 classes), zone (rows×cols) and x/y.
   The base model is pre-trained on a physics simulator: line-of-sight and wall/floor/ceiling reflections, body
   scattering and blocking, breathing, and radio impairments, over randomised rooms. It is then fine-tuned on your own
   recordings.
4. **Tracker.** Smoothing, presence hysteresis, an uncertainty radius and a trail. When the room is confidently empty,
   the baseline slowly adapts to drift.
5. **Vitals.** While the person is still, breathing rate comes from PCA over the CSI bands followed by a spectral peak
   search between 0.1 and 0.6 Hz.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
python -m scanphy serve --source sim --http-port 3000
```

MIT licence. three.js is bundled under its MIT licence (`scanphy/web/vendor/THREE_LICENSE`).
