"""
Detecção de Objetos em Tempo Real com YOLOv5 e Áudio

Este pacote fornece uma solução completa para detecção de objetos em tempo real
utilizando YOLOv5, OpenCV e Gradio com saída em áudio.
"""

__version__ = "1.0.0"
__author__ = "Rone Bragaglia"
__email__ = "ronbragaglia@gmail.com"

from .detection.detector import ObjectDetector
from .audio.speaker import Speaker
from .logging.logger import DetectionLogger
from .config.config import Config

__all__ = [
    "ObjectDetector",
    "Speaker",
    "DetectionLogger",
    "Config",
]
