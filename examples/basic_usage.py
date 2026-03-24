"""
Exemplo Básico de Uso do Sistema de Detecção de Objetos

Este exemplo demonstra como usar o sistema de detecção de objetos
de forma simples e direta.
"""

import sys
from pathlib import Path

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import Config
from src.detection.detector import ObjectDetector
from src.audio.speaker import Speaker
from src.logging.logger import DetectionLogger


def main():
    """Função principal do exemplo básico."""
    print("=" * 60)
    print("🎯 Exemplo Básico de Detecção de Objetos")
    print("=" * 60)
    print()
    
    # 1. Criar configuração
    print("1️⃣  Criando configuração...")
    config = Config(
        model_name="yolov5su.pt",
        confidence_threshold=0.5,
        audio_enabled=True,
        logging_enabled=True,
        save_images=True
    )
    print(f"✅ Configuração criada: {config}")
    print()
    
    # 2. Inicializar detector
    print("2️⃣  Inicializando detector...")
    detector = ObjectDetector(config)
    print(f"✅ Detector inicializado: {detector}")
    print(f"   Classes disponíveis: {len(detector.get_class_names())}")
    print()
    
    # 3. Inicializar speaker
    print("3️⃣  Inicializando speaker...")
    speaker = Speaker(config)
    print(f"✅ Speaker inicializado: {speaker}")
    print()
    
    # 4. Inicializar logger
    print("4️⃣  Inicializando logger...")
    logger = DetectionLogger(config)
    print(f"✅ Logger inicializado: {logger}")
    print()
    
    # 5. Detectar objetos em uma imagem
    print("5️⃣  Detectando objetos em uma imagem...")
    print("   (Este exemplo usa a webcam)")
    print()
    
    import cv2
    
    # Abrir webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível acessar a webcam!")
        return
    
    print("📷 Webcam aberta. Pressione 'q' para sair.")
    print()
    
    try:
        while True:
            # Capturar frame
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
                print(f"🔍 {detection.label} ({detection.confidence:.1%})")
                
                # Falar detecção
                speaker.speak_detection(
                    detection.label,
                    detection.confidence
                )
            
            # Logar detecções
            if detections:
                logger.log_detections(detections, frame)
            
            # Exibir frame
            cv2.imshow("Detecção de Objetos - Exemplo Básico", annotated_frame)
            
            # Verificar tecla 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupção pelo usuário.")
    
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        
        # Exibir estatísticas
        print()
        print("📊 Estatísticas de Detecção:")
        stats = logger.get_statistics()
        
        if stats:
            print(f"   Total de Detecções: {stats['total_detections']}")
            print(f"   Objetos Únicos: {stats['unique_objects']}")
            print(f"   Confiança Média: {stats['avg_confidence']:.2f}%")
            print(f"   Objetos Mais Comuns: {', '.join(stats['most_common'][:3])}")
        else:
            print("   Nenhuma detecção registrada.")
        
        print()
        print("=" * 60)
        print("✅ Exemplo concluído com sucesso!")
        print("=" * 60)


if __name__ == "__main__":
    main()
