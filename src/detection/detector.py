"""
Módulo de detecção de objetos usando YOLOv5.

Este módulo fornece uma classe para detecção de objetos em tempo real
utilizando o modelo YOLOv5 da Ultralytics.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics YOLO não está instalado. "
        "Instale com: pip install ultralytics"
    )

from ..config.config import Config


@dataclass
class Detection:
    """
    Representa uma detecção de objeto.
    
    Attributes:
        label: Rótulo do objeto detectado
        confidence: Confiança da detecção (0-1)
        bbox: Caixa delimitadora (x1, y1, x2, y2)
        frame: Frame da detecção (opcional)
    """
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    frame: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a detecção para um dicionário."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


class ObjectDetector:
    """
    Classe para detecção de objetos usando YOLOv5.
    
    Esta classe encapsula toda a lógica de detecção de objetos,
    incluindo carregamento do modelo, inferência e pós-processamento.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Inicializa o detector de objetos.
        
        Args:
            config: Configuração do detector. Se None, usa configuração padrão.
        """
        self.config = config or Config()
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Carregar o modelo
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo YOLOv5."""
        try:
            model_path = self.config.get_model_path()
            self.logger.info(f"Carregando modelo: {model_path}")
            
            # Verificar se o arquivo existe
            if not Path(model_path).exists():
                self.logger.warning(
                    f"Modelo não encontrado em {model_path}. "
                    "O modelo será baixado automaticamente."
                )
            
            self.model = YOLO(model_path)
            self.logger.info("Modelo carregado com sucesso!")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def detect(
        self,
        frame: np.ndarray,
        return_frame: bool = True
    ) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """
        Detecta objetos em um frame.
        
        Args:
            frame: Frame de entrada (imagem numpy array)
            return_frame: Se True, retorna o frame com bounding boxes desenhadas
            
        Returns:
            Tuple[List[Detection], Optional[np.ndarray]]: 
                Lista de detecções e frame anotado (se return_frame=True)
        """
        if self.model is None:
            raise RuntimeError("Modelo não foi carregado")
        
        # Executar inferência
        results = self.model(frame)
        
        detections = []
        annotated_frame = frame.copy() if return_frame else None
        
        # Processar resultados
        for result in results:
            for box in result.boxes:
                # Extrair informações da caixa
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                label = self.model.names[class_id]
                
                # Verificar confiança mínima
                if confidence < self.config.confidence_threshold:
                    continue
                
                # Verificar se a classe é permitida
                if not self.config.is_class_allowed(label):
                    continue
                
                # Criar detecção
                detection = Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    frame=frame.copy() if return_frame else None
                )
                detections.append(detection)
                
                # Desenhar bounding box se necessário
                if return_frame and annotated_frame is not None:
                    self._draw_detection(annotated_frame, detection)
        
        # Limitar número de detecções
        if len(detections) > self.config.max_detections:
            detections = detections[:self.config.max_detections]
        
        return detections, annotated_frame
    
    def _draw_detection(self, frame: np.ndarray, detection: Detection):
        """
        Desenha uma detecção no frame.
        
        Args:
            frame: Frame onde desenhar
            detection: Detecção a desenhar
        """
        x1, y1, x2, y2 = detection.bbox
        
        # Desenhar retângulo
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
        
        # Desenhar texto
        label_text = f"{detection.label} ({detection.confidence:.1%})"
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    def detect_from_image_path(
        self,
        image_path: str
    ) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """
        Detecta objetos em uma imagem de um arquivo.
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Tuple[List[Detection], Optional[np.ndarray]]: 
                Lista de detecções e frame anotado
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Não foi possível ler a imagem: {image_path}")
        
        return self.detect(frame, return_frame=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtém informações sobre o modelo.
        
        Returns:
            Dict[str, Any]: Informações do modelo
        """
        if self.model is None:
            return {}
        
        return {
            "model_name": self.config.model_name,
            "classes": self.model.names,
            "num_classes": len(self.model.names),
            "device": self.config.device,
        }
    
    def get_class_names(self) -> List[str]:
        """
        Obtém a lista de classes do modelo.
        
        Returns:
            List[str]: Lista de nomes das classes
        """
        if self.model is None:
            return []
        
        return list(self.model.names.values())
    
    def __repr__(self) -> str:
        """Representação string do detector."""
        return f"ObjectDetector(model={self.config.model_name}, confidence={self.config.confidence_threshold})"
