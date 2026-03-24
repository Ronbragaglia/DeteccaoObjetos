"""
Módulo de configuração para o sistema de detecção de objetos.

Este módulo gerencia todas as configurações do sistema, incluindo
parâmetros do modelo YOLO, configurações de áudio, logging, etc.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class Config:
    """
    Classe de configuração para o sistema de detecção de objetos.
    
    Attributes:
        model_name: Nome do modelo YOLO a ser usado
        confidence_threshold: Limiar de confiança para detecções
        max_detections: Número máximo de detecções por frame
        audio_enabled: Habilita/desabilita saída de áudio
        audio_language: Idioma para síntese de voz
        logging_enabled: Habilita/desabilita logging
        log_file: Caminho do arquivo de log
        save_images: Habilita/desabilita salvamento de imagens
        image_save_path: Diretório para salvar imagens detectadas
        webcam_index: Índice da webcam a ser usada
        gradio_share: Habilita compartilhamento via Gradio
        gradio_port: Porta para o servidor Gradio
        allowed_classes: Lista de classes permitidas (vazio = todas)
        excluded_classes: Lista de classes excluídas
    """
    
    # Configurações do Modelo
    model_name: str = "yolov5su.pt"
    confidence_threshold: float = 0.3
    max_detections: int = 100
    device: str = "cpu"  # "cpu" ou "cuda"
    
    # Configurações de Áudio
    audio_enabled: bool = True
    audio_language: str = "pt"
    audio_slow: bool = False
    temp_audio_path: str = "audio_temp.mp3"
    
    # Configurações de Logging
    logging_enabled: bool = True
    log_file: str = "detections_log.csv"
    log_level: str = "INFO"
    
    # Configurações de Imagens
    save_images: bool = True
    image_save_path: str = "detections_images"
    image_format: str = "jpg"
    image_quality: int = 95
    
    # Configurações de Webcam
    webcam_index: int = 0
    webcam_width: int = 640
    webcam_height: int = 480
    webcam_fps: int = 30
    
    # Configurações do Gradio
    gradio_share: bool = True
    gradio_port: int = 7860
    gradio_server_name: str = "0.0.0.0"
    
    # Filtros de Classes
    allowed_classes: List[str] = field(default_factory=list)
    excluded_classes: List[str] = field(default_factory=list)
    
    # Outros
    verbose: bool = True
    debug_mode: bool = False
    
    def __post_init__(self):
        """Inicialização pós-criação da configuração."""
        # Criar diretórios necessários
        Path(self.image_save_path).mkdir(parents=True, exist_ok=True)
        
        # Validar configurações
        self._validate()
    
    def _validate(self):
        """Valida as configurações."""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold deve estar entre 0 e 1")
        
        if self.max_detections <= 0:
            raise ValueError("max_detections deve ser maior que 0")
        
        if self.image_quality < 1 or self.image_quality > 100:
            raise ValueError("image_quality deve estar entre 1 e 100")
        
        if self.webcam_fps <= 0:
            raise ValueError("webcam_fps deve ser maior que 0")
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Cria uma configuração a partir de variáveis de ambiente.
        
        Returns:
            Config: Instância de configuração carregada do ambiente
        """
        return cls(
            model_name=os.getenv("MODEL_NAME", "yolov5su.pt"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.3")),
            max_detections=int(os.getenv("MAX_DETECTIONS", "100")),
            device=os.getenv("DEVICE", "cpu"),
            audio_enabled=os.getenv("AUDIO_ENABLED", "true").lower() == "true",
            audio_language=os.getenv("AUDIO_LANGUAGE", "pt"),
            logging_enabled=os.getenv("LOGGING_ENABLED", "true").lower() == "true",
            log_file=os.getenv("LOG_FILE", "detections_log.csv"),
            save_images=os.getenv("SAVE_IMAGES", "true").lower() == "true",
            image_save_path=os.getenv("IMAGE_SAVE_PATH", "detections_images"),
            webcam_index=int(os.getenv("WEBCAM_INDEX", "0")),
            gradio_share=os.getenv("GRADIO_SHARE", "true").lower() == "true",
            gradio_port=int(os.getenv("GRADIO_PORT", "7860")),
            verbose=os.getenv("VERBOSE", "true").lower() == "true",
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """
        Cria uma configuração a partir de um dicionário.
        
        Args:
            config_dict: Dicionário com configurações
            
        Returns:
            Config: Instância de configuração
        """
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """
        Converte a configuração para um dicionário.
        
        Returns:
            dict: Dicionário com configurações
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def is_class_allowed(self, class_name: str) -> bool:
        """
        Verifica se uma classe é permitida.
        
        Args:
            class_name: Nome da classe
            
        Returns:
            bool: True se a classe é permitida, False caso contrário
        """
        # Se há classes permitidas específicas, verificar
        if self.allowed_classes:
            return class_name in self.allowed_classes
        
        # Se a classe está na lista de excluídas, não permitir
        if class_name in self.excluded_classes:
            return False
        
        # Caso contrário, permitir
        return True
    
    def get_model_path(self) -> str:
        """
        Obtém o caminho completo do modelo.
        
        Returns:
            str: Caminho do modelo
        """
        # Se o caminho já é absoluto, retornar como está
        if os.path.isabs(self.model_name):
            return self.model_name
        
        # Caso contrário, assumir que está no diretório de modelos
        return os.path.join("models", self.model_name)
    
    def __repr__(self) -> str:
        """Representação string da configuração."""
        return f"Config(model={self.model_name}, confidence={self.confidence_threshold}, audio={self.audio_enabled})"
