"""WaveSense 3D - WiFi CSI human sensing and indoor localisation for ESP32-S3 + NVIDIA Jetson.

Pipeline:
    ESP32-S3 nodes (CSI over UDP) -> wavesense.server (Jetson) -> DSP + neural model -> 3D dashboard
"""

__version__ = "2.0.0"
APP_NAME = "WaveSense 3D"
