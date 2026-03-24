"""
Testes para o módulo de detecção.
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import Config
from src.detection.detector import ObjectDetector, Detection


class TestDetection:
    """Testes para a classe Detection."""
    
    def test_detection_creation(self):
        """Testa criação de uma detecção."""
        detection = Detection(
            label="person",
            confidence=0.85,
            bbox=(10, 20, 100, 200)
        )
        
        assert detection.label == "person"
        assert detection.confidence == 0.85
        assert detection.bbox == (10, 20, 100, 200)
        assert detection.frame is None
    
    def test_detection_to_dict(self):
        """Testa conversão de detecção para dicionário."""
        detection = Detection(
            label="car",
            confidence=0.92,
            bbox=(50, 60, 150, 160)
        )
        
        detection_dict = detection.to_dict()
        
        assert detection_dict["label"] == "car"
        assert detection_dict["confidence"] == 0.92
        assert detection_dict["bbox"] == (50, 60, 150, 160)
        assert "frame" not in detection_dict


@pytest.mark.unit
class TestObjectDetector:
    """Testes para a classe ObjectDetector."""
    
    @pytest.fixture
    def config(self):
        """Fixture para configuração."""
        return Config(
            model_name="yolov5su.pt",
            confidence_threshold=0.5,
            max_detections=10
        )
    
    @pytest.fixture
    def detector(self, config):
        """Fixture para detector."""
        return ObjectDetector(config)
    
    def test_detector_initialization(self, detector):
        """Testa inicialização do detector."""
        assert detector is not None
        assert detector.model is not None
        assert detector.config is not None
    
    def test_detect_empty_frame(self, detector):
        """Testa detecção em frame vazio."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections, annotated_frame = detector.detect(frame)
        
        assert isinstance(detections, list)
        assert len(detections) == 0
        assert annotated_frame is not None
    
    def test_detect_with_return_frame(self, detector):
        """Testa detecção com retorno de frame anotado."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections, annotated_frame = detector.detect(frame, return_frame=True)
        
        assert annotated_frame is not None
        assert annotated_frame.shape == frame.shape
    
    def test_detect_without_return_frame(self, detector):
        """Testa detecção sem retorno de frame anotado."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections, annotated_frame = detector.detect(frame, return_frame=False)
        
        assert annotated_frame is None
    
    def test_max_detections_limit(self, config):
        """Testa limite máximo de detecções."""
        config.max_detections = 5
        detector = ObjectDetector(config)
        
        # Criar frame com múltiplos objetos
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections, _ = detector.detect(frame)
        
        # Verificar se o limite é respeitado
        # (pode não haver detecções, então não podemos garantir)
        if len(detections) > 0:
            assert len(detections) <= config.max_detections
    
    def test_confidence_threshold(self, config):
        """Testa limiar de confiança."""
        config.confidence_threshold = 0.9
        detector = ObjectDetector(config)
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections, _ = detector.detect(frame)
        
        # Verificar se todas as detecções têm confiança acima do limiar
        for detection in detections:
            assert detection.confidence >= config.confidence_threshold
    
    def test_get_model_info(self, detector):
        """Testa obtenção de informações do modelo."""
        info = detector.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "classes" in info
        assert "num_classes" in info
        assert "device" in info
    
    def test_get_class_names(self, detector):
        """Testa obtenção de nomes das classes."""
        class_names = detector.get_class_names()
        
        assert isinstance(class_names, list)
        assert len(class_names) > 0
        assert "person" in class_names
    
    def test_repr(self, detector):
        """Testa representação string do detector."""
        repr_str = repr(detector)
        
        assert "ObjectDetector" in repr_str
        assert "yolov5su.pt" in repr_str


@pytest.mark.integration
class TestObjectDetectorIntegration:
    """Testes de integração para o detector."""
    
    @pytest.fixture
    def config(self):
        """Fixture para configuração."""
        return Config(
            model_name="yolov5su.pt",
            confidence_threshold=0.3
        )
    
    @pytest.fixture
    def detector(self, config):
        """Fixture para detector."""
        return ObjectDetector(config)
    
    def test_detect_real_image(self, detector):
        """Testa detecção em uma imagem real."""
        # Criar uma imagem simples com um quadrado
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        
        detections, annotated_frame = detector.detect(frame)
        
        assert isinstance(detections, list)
        assert annotated_frame is not None
    
    def test_detect_multiple_frames(self, detector):
        """Testa detecção em múltiplos frames."""
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        for frame in frames:
            detections, _ = detector.detect(frame)
            assert isinstance(detections, list)
