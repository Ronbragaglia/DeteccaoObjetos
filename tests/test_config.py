"""
Testes para o módulo de configuração.
"""

import pytest
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import Config


class TestConfig:
    """Testes para a classe Config."""
    
    def test_default_config(self):
        """Testa criação de configuração padrão."""
        config = Config()
        
        assert config.model_name == "yolov5su.pt"
        assert config.confidence_threshold == 0.3
        assert config.max_detections == 100
        assert config.audio_enabled is True
        assert config.logging_enabled is True
        assert config.save_images is True
    
    def test_custom_config(self):
        """Testa criação de configuração personalizada."""
        config = Config(
            model_name="yolov8n.pt",
            confidence_threshold=0.7,
            max_detections=50,
            audio_enabled=False
        )
        
        assert config.model_name == "yolov8n.pt"
        assert config.confidence_threshold == 0.7
        assert config.max_detections == 50
        assert config.audio_enabled is False
    
    def test_invalid_confidence_threshold(self):
        """Testa validação de limiar de confiança inválido."""
        with pytest.raises(ValueError):
            Config(confidence_threshold=1.5)
        
        with pytest.raises(ValueError):
            Config(confidence_threshold=-0.1)
    
    def test_invalid_max_detections(self):
        """Testa validação de número máximo de detecções inválido."""
        with pytest.raises(ValueError):
            Config(max_detections=0)
        
        with pytest.raises(ValueError):
            Config(max_detections=-10)
    
    def test_invalid_image_quality(self):
        """Testa validação de qualidade de imagem inválida."""
        with pytest.raises(ValueError):
            Config(image_quality=0)
        
        with pytest.raises(ValueError):
            Config(image_quality=150)
    
    def test_from_env(self):
        """Testa carregamento de configuração do ambiente."""
        # Definir variáveis de ambiente
        os.environ["MODEL_NAME"] = "yolov8n.pt"
        os.environ["CONFIDENCE_THRESHOLD"] = "0.8"
        os.environ["AUDIO_ENABLED"] = "false"
        
        try:
            config = Config.from_env()
            
            assert config.model_name == "yolov8n.pt"
            assert config.confidence_threshold == 0.8
            assert config.audio_enabled is False
        finally:
            # Limpar variáveis de ambiente
            del os.environ["MODEL_NAME"]
            del os.environ["CONFIDENCE_THRESHOLD"]
            del os.environ["AUDIO_ENABLED"]
    
    def test_from_dict(self):
        """Testa criação de configuração a partir de dicionário."""
        config_dict = {
            "model_name": "yolov8n.pt",
            "confidence_threshold": 0.9,
            "audio_enabled": False
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.model_name == "yolov8n.pt"
        assert config.confidence_threshold == 0.9
        assert config.audio_enabled is False
    
    def test_to_dict(self):
        """Testa conversão de configuração para dicionário."""
        config = Config(
            model_name="yolov8n.pt",
            confidence_threshold=0.7
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["model_name"] == "yolov8n.pt"
        assert config_dict["confidence_threshold"] == 0.7
        assert isinstance(config_dict, dict)
    
    def test_is_class_allowed_no_filters(self):
        """Testa verificação de classe permitida sem filtros."""
        config = Config()
        
        assert config.is_class_allowed("person") is True
        assert config.is_class_allowed("car") is True
        assert config.is_class_allowed("dog") is True
    
    def test_is_class_allowed_with_allowed_classes(self):
        """Testa verificação de classe permitida com classes permitidas."""
        config = Config(allowed_classes=["person", "car"])
        
        assert config.is_class_allowed("person") is True
        assert config.is_class_allowed("car") is True
        assert config.is_class_allowed("dog") is False
    
    def test_is_class_allowed_with_excluded_classes(self):
        """Testa verificação de classe permitida com classes excluídas."""
        config = Config(excluded_classes=["dog", "cat"])
        
        assert config.is_class_allowed("person") is True
        assert config.is_class_allowed("dog") is False
        assert config.is_class_allowed("cat") is False
    
    def test_get_model_path_absolute(self):
        """Testa obtenção de caminho do modelo absoluto."""
        config = Config(model_name="/absolute/path/to/model.pt")
        
        path = config.get_model_path()
        
        assert path == "/absolute/path/to/model.pt"
    
    def test_get_model_path_relative(self):
        """Testa obtenção de caminho do modelo relativo."""
        config = Config(model_name="yolov5su.pt")
        
        path = config.get_model_path()
        
        assert path == "models/yolov5su.pt"
    
    def test_repr(self):
        """Testa representação string da configuração."""
        config = Config(
            model_name="yolov8n.pt",
            confidence_threshold=0.7
        )
        
        repr_str = repr(config)
        
        assert "yolov8n.pt" in repr_str
        assert "0.7" in repr_str
