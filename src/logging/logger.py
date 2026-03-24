"""
Módulo de logging para detecções de objetos.

Este módulo fornece uma classe para registro de detecções em CSV
e salvamento de imagens detectadas.
"""

import os
import cv2
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import pandas as pd

from ..config.config import Config
from ..detection.detector import Detection


class DetectionLogger:
    """
    Classe para logging de detecções de objetos.
    
    Esta classe encapsula toda a lógica de registro de detecções,
    incluindo salvamento em CSV e armazenamento de imagens.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Inicializa o logger de detecções.
        
        Args:
            config: Configuração do logger. Se None, usa configuração padrão.
        """
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # Criar diretórios necessários
        self._setup_directories()
    
    def _setup_directories(self):
        """Cria os diretórios necessários para logging."""
        # Criar diretório para imagens
        if self.config.save_images:
            Path(self.config.image_save_path).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Diretório de imagens: {self.config.image_save_path}")
    
    def log_detections(
        self,
        detections: List[Detection],
        frame: Optional[cv2.typing.MatLike] = None
    ) -> bool:
        """
        Registra uma lista de detecções.
        
        Args:
            detections: Lista de detecções a registrar
            frame: Frame original (opcional, para salvar imagens)
            
        Returns:
            bool: True se o logging foi bem-sucedido, False caso contrário
        """
        if not self.config.logging_enabled:
            self.logger.debug("Logging desabilitado, ignorando registro")
            return False
        
        if not detections:
            self.logger.debug("Nenhuma detecção para registrar")
            return False
        
        try:
            timestamp = datetime.now()
            log_entries = []
            
            for detection in detections:
                # Criar entrada de log
                log_entry = {
                    "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    "label": detection.label,
                    "confidence": f"{detection.confidence:.2%}",
                    "bbox": str(detection.bbox),
                }
                log_entries.append(log_entry)
                
                # Salvar imagem se habilitado
                if self.config.save_images and frame is not None:
                    self._save_detection_image(
                        detection,
                        frame,
                        timestamp
                    )
            
            # Salvar no CSV
            self._save_to_csv(log_entries)
            
            self.logger.info(f"{len(detections)} detecções registradas")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao registrar detecções: {e}")
            return False
    
    def _save_detection_image(
        self,
        detection: Detection,
        frame: cv2.typing.MatLike,
        timestamp: datetime
    ):
        """
        Salva uma imagem de detecção.
        
        Args:
            detection: Detecção a salvar
            frame: Frame original
            timestamp: Timestamp da detecção
        """
        try:
            # Criar nome do arquivo
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp_str}_{detection.label}.{self.config.image_format}"
            filepath = os.path.join(self.config.image_save_path, filename)
            
            # Recortar a área da detecção
            x1, y1, x2, y2 = detection.bbox
            cropped = frame[y1:y2, x1:x2]
            
            # Salvar imagem
            cv2.imwrite(
                filepath,
                cropped,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.config.image_quality]
            )
            
            self.logger.debug(f"Imagem salva: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar imagem: {e}")
    
    def _save_to_csv(self, log_entries: List[dict]):
        """
        Salva entradas de log em CSV.
        
        Args:
            log_entries: Lista de entradas de log
        """
        try:
            # Criar DataFrame
            df = pd.DataFrame(log_entries)
            
            # Verificar se o arquivo já existe
            file_exists = os.path.exists(self.config.log_file)
            
            # Salvar no CSV
            df.to_csv(
                self.config.log_file,
                mode='a',
                header=not file_exists,
                index=False
            )
            
            self.logger.debug(f"Log salvo em: {self.config.log_file}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar CSV: {e}")
            raise
    
    def get_log_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Lê o log de detecções como um DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame com o log, ou None se não existir
        """
        try:
            if not os.path.exists(self.config.log_file):
                self.logger.warning(f"Arquivo de log não encontrado: {self.config.log_file}")
                return None
            
            df = pd.read_csv(self.config.log_file)
            self.logger.info(f"Log carregado: {len(df)} entradas")
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao ler log: {e}")
            return None
    
    def get_statistics(self) -> Optional[dict]:
        """
        Calcula estatísticas das detecções.
        
        Returns:
            Optional[dict]: Dicionário com estatísticas, ou None se não houver dados
        """
        df = self.get_log_dataframe()
        
        if df is None or df.empty:
            return None
        
        try:
            stats = {
                "total_detections": len(df),
                "unique_objects": df["label"].nunique(),
                "most_common": df["label"].mode().tolist(),
                "avg_confidence": df["confidence"].str.rstrip('%').astype(float).mean(),
                "date_range": {
                    "start": df["timestamp"].min(),
                    "end": df["timestamp"].max(),
                },
            }
            
            # Adicionar contagem por objeto
            stats["object_counts"] = df["label"].value_counts().to_dict()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular estatísticas: {e}")
            return None
    
    def clear_log(self) -> bool:
        """
        Limpa o arquivo de log.
        
        Returns:
            bool: True se o log foi limpo com sucesso
        """
        try:
            if os.path.exists(self.config.log_file):
                os.remove(self.config.log_file)
                self.logger.info(f"Log removido: {self.config.log_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao limpar log: {e}")
            return False
    
    def export_log(self, output_path: str, format: str = "csv") -> bool:
        """
        Exporta o log para um arquivo.
        
        Args:
            output_path: Caminho do arquivo de saída
            format: Formato de exportação (csv, json, excel)
            
        Returns:
            bool: True se a exportação foi bem-sucedida
        """
        df = self.get_log_dataframe()
        
        if df is None:
            self.logger.warning("Não há dados para exportar")
            return False
        
        try:
            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "json":
                df.to_json(output_path, orient="records", indent=2)
            elif format == "excel":
                df.to_excel(output_path, index=False)
            else:
                self.logger.error(f"Formato não suportado: {format}")
                return False
            
            self.logger.info(f"Log exportado para: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar log: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """
        Verifica se o logging está habilitado.
        
        Returns:
            bool: True se o logging está habilitado
        """
        return self.config.logging_enabled
    
    def set_enabled(self, enabled: bool):
        """
        Habilita ou desabilita o logging.
        
        Args:
            enabled: True para habilitar, False para desabilitar
        """
        self.config.logging_enabled = enabled
        self.logger.info(f"Logging {'habilitado' if enabled else 'desabilitado'}")
    
    def __repr__(self) -> str:
        """Representação string do logger."""
        return f"DetectionLogger(enabled={self.config.logging_enabled}, log_file={self.config.log_file})"
