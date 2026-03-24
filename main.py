"""
Sistema de Detecção de Objetos em Tempo Real com YOLOv5 e Áudio

Este é o ponto de entrada principal para o sistema de detecção de objetos.
"""

import logging
import sys
from pathlib import Path

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.config import Config
from src.detection.detector import ObjectDetector
from src.audio.speaker import Speaker
from src.logging.logger import DetectionLogger
from src.interface.gradio_interface import GradioInterface


def setup_logging(config: Config):
    """
    Configura o sistema de logging.
    
    Args:
        config: Configuração do sistema
    """
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )


def main():
    """Função principal do sistema."""
    print("🎯 Iniciando Sistema de Detecção de Objetos...")
    print("=" * 60)
    
    # Carregar configuração
    print("📋 Carregando configurações...")
    config = Config.from_env()
    print(f"✅ Configurações carregadas: {config}")
    print()
    
    # Configurar logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Inicializar componentes
    print("🔧 Inicializando componentes...")
    
    try:
        # Detector
        print("  - Carregando modelo YOLOv5...")
        detector = ObjectDetector(config)
        model_info = detector.get_model_info()
        print(f"  ✅ Modelo carregado: {model_info['model_name']}")
        print(f"     Classes: {model_info['num_classes']}")
        print()
        
        # Speaker
        print("  - Inicializando speaker...")
        speaker = Speaker(config)
        print(f"  ✅ Speaker inicializado: {speaker}")
        print()
        
        # Logger
        print("  - Inicializando logger...")
        logger_obj = DetectionLogger(config)
        print(f"  ✅ Logger inicializado: {logger_obj}")
        print()
        
    except Exception as e:
        logger.error(f"Erro ao inicializar componentes: {e}")
        print(f"❌ Erro ao inicializar componentes: {e}")
        sys.exit(1)
    
    # Criar interface
    print("🖥️  Criando interface Gradio...")
    interface = GradioInterface(
        config=config,
        detector=detector,
        speaker=speaker,
        logger=logger_obj
    )
    print(f"✅ Interface criada: {interface}")
    print()
    
    # Exibir informações
    print("=" * 60)
    print("📊 Informações do Sistema:")
    print(f"  Modelo: {config.model_name}")
    print(f"  Confiança Mínima: {config.confidence_threshold:.0%}")
    print(f"  Áudio: {'Habilitado' if config.audio_enabled else 'Desabilitado'}")
    print(f"  Logging: {'Habilitado' if config.logging_enabled else 'Desabilitado'}")
    print(f"  Salvar Imagens: {'Sim' if config.save_images else 'Não'}")
    print("=" * 60)
    print()
    
    # Iniciar interface
    print("🚀 Iniciando interface...")
    print("Acesse a interface no navegador para começar a detecção!")
    print()
    
    try:
        interface.launch()
    except KeyboardInterrupt:
        print("\n\n⏹️  Sistema encerrado pelo usuário.")
    except Exception as e:
        logger.error(f"Erro ao executar interface: {e}")
        print(f"❌ Erro ao executar interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
