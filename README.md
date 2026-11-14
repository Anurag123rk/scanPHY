# scanPHY

WiFi CSI human sensing and indoor localisation using 3× ESP32-S3 + NVIDIA Jetson Orin.
A standalone alternative to RuView, with a built-in auto-training pipeline.

> **Work in progress.** Done: Python core (protocol, features, physics simulator, NumPy inference runtime,
> PyTorch trainer with ONNX export, dataset recorder, sensing engine, vitals, auto-trainer, aiohttp server).
> Still to do: dashboard UI (monochrome), ESP32-S3 firmware, flash/provision tools, Jetson setup scripts,
> the pretrained base model, and tests.

Ports: UDP 5005 (CSI in), HTTP 3000 (dashboard + `/ws`), WebSocket 3001.
