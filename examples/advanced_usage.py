"""
Exemplo Avançado de Uso do Sistema de Detecção de Objetos

Este exemplo demonstra recursos avançados do sistema, incluindo:
- Filtro de classes específicas
- Exportação de logs
- Análise de estatísticas
- Personalização de configurações
"""

import sys
from pathlib import Path

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import Config
from src.detection.detector import ObjectDetector
from src.audio.speaker import Speaker
from src.logging.logger import DetectionLogger
import cv2
import time


def advanced_detection_example():
    """Exemplo avançado com filtros e personalização."""
    print("=" * 60)
    print("🎯 Exemplo Avançado de Detecção de Objetos")
    print("=" * 60)
    print()
    
    # 1. Criar configuração personalizada
    print("1️⃣  Criando configuração personalizada...")
    config = Config(
        model_name="yolov5su.pt",
        confidence_threshold=0.6,  # Confiança mais alta
        max_detections=50,
        audio_enabled=True,
        audio_language="pt",
        logging_enabled=True,
        save_images=True,
        image_quality=90,
        verbose=True
    )
    
    # Filtrar apenas classes específicas
    config.allowed_classes = ["person", "cell phone", "laptop", "tv"]
    print(f"✅ Configuração criada")
    print(f"   Classes permitidas: {config.allowed_classes}")
    print()
    
    # 2. Inicializar componentes
    print("2️⃣  Inicializando componentes...")
    detector = ObjectDetector(config)
    speaker = Speaker(config)
    logger = DetectionLogger(config)
    print("✅ Componentes inicializados")
    print()
    
    # 3. Detectar objetos em múltiplas imagens
    print("3️⃣  Detectando objetos em tempo real...")
    print("   (Pressione 'q' para sair, 's' para salvar estatísticas)")
    print()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível acessar a webcam!")
        return
    
    detection_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detectar objetos
            detections, annotated_frame = detector.detect(
                frame,
                return_frame=True
            )
            
            # Processar detecções
            for detection in detections:
                detection_count += 1
                
                # Exibir informações
                print(f"🔍 [{detection_count}] {detection.label} - "
                      f"Confiança: {detection.confidence:.1%}")
                
                # Falar apenas para confiança alta
                if detection.confidence > 0.8:
                    speaker.speak_detection(
                        detection.label,
                        detection.confidence
                    )
            
            # Logar detecções
            if detections:
                logger.log_detections(detections, frame)
            
            # Exibir informações na tela
            info_text = f"Detections: {detection_count} | "
            info_text += f"FPS: {1.0 / (time.time() - start_time + 0.001):.1f}"
            
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Exibir frame
            cv2.imshow("Detecção Avançada - YOLO", annotated_frame)
            
            # Verificar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Salvar estatísticas
                print("\n📊 Salvando estatísticas...")
                save_statistics(logger)
            
            start_time = time.time()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupção pelo usuário.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Exibir estatísticas finais
        print("\n📊 Estatísticas Finais:")
        display_statistics(logger)
        print()
        print("=" * 60)
        print("✅ Exemplo avançado concluído!")
        print("=" * 60)


def save_statistics(logger: DetectionLogger):
    """
    Salva estatísticas em diferentes formatos.
    
    Args:
        logger: Instância do logger de detecções
    """
    stats = logger.get_statistics()
    
    if stats is None:
        print("❌ Nenhuma detecção registrada.")
        return
    
    # Exportar para CSV
    logger.export_log("detections_export.csv", format="csv")
    print("✅ Log exportado para: detections_export.csv")
    
    # Exportar para JSON
    logger.export_log("detections_export.json", format="json")
    print("✅ Log exportado para: detections_export.json")
    
    # Exibir estatísticas
    print(f"\n📈 Estatísticas:")
    print(f"   Total de Detecções: {stats['total_detections']}")
    print(f"   Objetos Únicos: {stats['unique_objects']}")
    print(f"   Confiança Média: {stats['avg_confidence']:.2f}%")
    
    print(f"\n📊 Contagem por Objeto:")
    for obj, count in stats['object_counts'].items():
        print(f"   - {obj}: {count}")


def display_statistics(logger: DetectionLogger):
    """
    Exibe estatísticas de detecção de forma formatada.
    
    Args:
        logger: Instância do logger de detecções
    """
    stats = logger.get_statistics()
    
    if stats is None:
        print("❌ Nenhuma detecção registrada.")
        return
    
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO DE DETECÇÕES")
    print(f"{'='*60}")
    
    print(f"\n📈 Resumo Geral:")
    print(f"   Total de Detecções: {stats['total_detections']}")
    print(f"   Objetos Únicos: {stats['unique_objects']}")
    print(f"   Confiança Média: {stats['avg_confidence']:.2f}%")
    
    print(f"\n📊 Objetos Mais Comuns:")
    for i, (obj, count) in enumerate(stats['object_counts'].items(), 1):
        percentage = (count / stats['total_detections']) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {i:2d}. {obj:20s} {count:4d} ({percentage:5.1f}%) {bar}")
    
    print(f"\n📅 Período de Detecção:")
    print(f"   Início: {stats['date_range']['start']}")
    print(f"   Fim:    {stats['date_range']['end']}")
    
    print(f"\n{'='*60}")


def custom_configuration_example():
    """Exemplo de configuração personalizada."""
    print("\n" + "="*60)
    print("⚙️  Exemplo de Configuração Personalizada")
    print("="*60)
    print()
    
    # Configuração a partir de variáveis de ambiente
    print("1️⃣  Carregando configuração do ambiente...")
    config = Config.from_env()
    print(f"✅ Configuração carregada: {config}")
    print()
    
    # Configuração a partir de dicionário
    print("2️⃣  Criando configuração a partir de dicionário...")
    config_dict = {
        "model_name": "yolov5su.pt",
        "confidence_threshold": 0.7,
        "audio_enabled": False,
        "logging_enabled": True,
        "allowed_classes": ["person", "car"]
    }
    config = Config.from_dict(config_dict)
    print(f"✅ Configuração criada: {config}")
    print()
    
    # Modificar configuração em tempo de execução
    print("3️⃣  Modificando configuração em tempo de execução...")
    config.confidence_threshold = 0.8
    config.audio_enabled = True
    print(f"✅ Configuração atualizada:")
    print(f"   Confiança: {config.confidence_threshold}")
    print(f"   Áudio: {config.audio_enabled}")
    print()
    
    # Exportar configuração
    print("4️⃣  Exportando configuração...")
    config_dict = config.to_dict()
    print(f"✅ Configuração exportada:")
    for key, value in config_dict.items():
        print(f"   {key}: {value}")
    print()


def main():
    """Função principal do exemplo avançado."""
    print("\n" + "🚀"*30)
    print("SISTEMA DE DETECÇÃO DE OBJETOS - EXEMPLO AVANÇADO")
    print("🚀"*30)
    print()
    
    # Exemplo de configuração personalizada
    custom_configuration_example()
    
    # Exemplo avançado de detecção
    advanced_detection_example()
    
    print("\n" + "="*60)
    print("🎉 Todos os exemplos avançados concluídos!")
    print("="*60)
    print()


if __name__ == "__main__":
    main()
