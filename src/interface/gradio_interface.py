"""
Módulo de interface Gradio para detecção de objetos.

Este módulo fornece uma interface de usuário interativa usando Gradio
para detecção de objetos em tempo real e em imagens.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
import threading

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio não está instalado. "
        "Instale com: pip install gradio"
    )

from ..config.config import Config
from ..detection.detector import ObjectDetector
from ..audio.speaker import Speaker
from ..logging.logger import DetectionLogger


class GradioInterface:
    """
    Classe para criação de interface Gradio para detecção de objetos.
    
    Esta classe encapsula toda a lógica de criação e gerenciamento
    da interface de usuário usando Gradio.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        detector: Optional[ObjectDetector] = None,
        speaker: Optional[Speaker] = None,
        logger: Optional[DetectionLogger] = None
    ):
        """
        Inicializa a interface Gradio.
        
        Args:
            config: Configuração da interface
            detector: Detector de objetos
            speaker: Speaker para áudio
            logger: Logger para detecções
        """
        self.config = config or Config()
        self.detector = detector or ObjectDetector(self.config)
        self.speaker = speaker or Speaker(self.config)
        self.logger = logger or DetectionLogger(self.config)
        
        self.logger_interface = logging.getLogger(__name__)
        self.detection_thread = None
        self.is_detecting = False
        
        # Criar interface
        self.interface = self._create_interface()
    
    def _create_interface(self) -> gr.Interface:
        """
        Cria a interface Gradio.
        
        Returns:
            gr.Interface: Interface Gradio criada
        """
        with gr.Blocks(
            title="🎯 Detecção de Objetos com YOLOv5 e Áudio",
            theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown(
                """
                # 🎯 Detecção de Objetos com YOLOv5 e Áudio
                
                Sistema de detecção de objetos em tempo real usando YOLOv5, 
                OpenCV e Gradio com saída em áudio.
                """
            )
            
            with gr.Tabs():
                # Aba de detecção em imagens
                with gr.Tab("📷 Detecção em Imagens"):
                    with gr.Row():
                        image_input = gr.Image(
                            type="numpy",
                            label="Carregar Imagem"
                        )
                        image_output = gr.Image(
                            type="numpy",
                            label="Resultado"
                        )
                    
                    detect_btn = gr.Button(
                        "🔍 Detectar Objetos",
                        variant="primary"
                    )
                    
                    detect_btn.click(
                        fn=self._detect_objects_image,
                        inputs=image_input,
                        outputs=image_output
                    )
                
                # Aba de detecção em tempo real
                with gr.Tab("📹 Detecção em Tempo Real"):
                    status_text = gr.Textbox(
                        label="Status",
                        value="Aguardando início...",
                        interactive=False
                    )
                    
                    with gr.Row():
                        start_btn = gr.Button(
                            "▶️ Iniciar Detecção",
                            variant="primary"
                        )
                        stop_btn = gr.Button(
                            "⏹️ Parar Detecção",
                            variant="stop"
                        )
                    
                    start_btn.click(
                        fn=self._start_detection,
                        outputs=status_text
                    )
                    
                    stop_btn.click(
                        fn=self._stop_detection,
                        outputs=status_text
                    )
                
                # Aba de configurações
                with gr.Tab("⚙️ Configurações"):
                    with gr.Row():
                        confidence_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=self.config.confidence_threshold,
                            step=0.05,
                            label="Limiar de Confiança"
                        )
                        
                        audio_checkbox = gr.Checkbox(
                            value=self.config.audio_enabled,
                            label="Habilitar Áudio"
                        )
                    
                    with gr.Row():
                        logging_checkbox = gr.Checkbox(
                            value=self.config.logging_enabled,
                            label="Habilitar Logging"
                        )
                        
                        save_images_checkbox = gr.Checkbox(
                            value=self.config.save_images,
                            label="Salvar Imagens"
                        )
                    
                    apply_btn = gr.Button(
                        "✅ Aplicar Configurações",
                        variant="primary"
                    )
                    
                    apply_btn.click(
                        fn=self._apply_config,
                        inputs=[
                            confidence_slider,
                            audio_checkbox,
                            logging_checkbox,
                            save_images_checkbox
                        ],
                        outputs=status_text
                    )
                
                # Aba de estatísticas
                with gr.Tab("📊 Estatísticas"):
                    stats_text = gr.Textbox(
                        label="Estatísticas de Detecção",
                        lines=10,
                        interactive=False
                    )
                    
                    refresh_stats_btn = gr.Button(
                        "🔄 Atualizar Estatísticas"
                    )
                    
                    refresh_stats_btn.click(
                        fn=self._get_statistics,
                        outputs=stats_text
                    )
            
            gr.Markdown(
                """
                ---
                
                **Desenvolvido por:** Rone Bragaglia  
                **Versão:** 1.0.0
                """
            )
        
        return interface
    
    def _detect_objects_image(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta objetos em uma imagem.
        
        Args:
            image: Imagem de entrada
            
        Returns:
            np.ndarray: Imagem com bounding boxes desenhadas
        """
        if image is None:
            return image
        
        try:
            detections, annotated_frame = self.detector.detect(
                image,
                return_frame=True
            )
            
            # Falar detecções
            for detection in detections:
                self.speaker.speak_detection(
                    detection.label,
                    detection.confidence
                )
            
            # Logar detecções
            self.logger.log_detections(detections, image)
            
            return annotated_frame
            
        except Exception as e:
            self.logger_interface.error(f"Erro ao detectar objetos: {e}")
            return image
    
    def _start_detection(self) -> str:
        """
        Inicia a detecção em tempo real.
        
        Returns:
            str: Mensagem de status
        """
        if self.is_detecting:
            return "⚠️ Detecção já está em andamento!"
        
        try:
            self.is_detecting = True
            self.detection_thread = threading.Thread(
                target=self._run_detection_loop,
                daemon=True
            )
            self.detection_thread.start()
            
            return "✅ Detecção iniciada! Pressione 'q' na janela OpenCV para parar."
            
        except Exception as e:
            self.logger_interface.error(f"Erro ao iniciar detecção: {e}")
            self.is_detecting = False
            return f"❌ Erro ao iniciar detecção: {e}"
    
    def _stop_detection(self) -> str:
        """
        Para a detecção em tempo real.
        
        Returns:
            str: Mensagem de status
        """
        if not self.is_detecting:
            return "⚠️ Detecção não está em andamento!"
        
        self.is_detecting = False
        return "⏹️ Detecção parada!"
    
    def _run_detection_loop(self):
        """
        Executa o loop de detecção em tempo real.
        """
        try:
            cap = cv2.VideoCapture(self.config.webcam_index)
            
            if not cap.isOpened():
                self.logger_interface.error(
                    "Não foi possível acessar a webcam!"
                )
                self.is_detecting = False
                return
            
            # Configurar webcam
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.webcam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.webcam_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.webcam_fps)
            
            self.logger_interface.info("Iniciando detecção em tempo real...")
            
            while self.is_detecting:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Detectar objetos
                detections, annotated_frame = self.detector.detect(
                    frame,
                    return_frame=True
                )
                
                # Falar detecções
                for detection in detections:
                    self.speaker.speak_detection(
                        detection.label,
                        detection.confidence
                    )
                
                # Logar detecções
                self.logger.log_detections(detections, frame)
                
                # Exibir frame
                cv2.imshow("Detecção de Objetos - YOLO", annotated_frame)
                
                # Verificar tecla 'q' para sair
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.is_detecting = False
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.logger_interface.error(f"Erro no loop de detecção: {e}")
        finally:
            self.is_detecting = False
    
    def _apply_config(
        self,
        confidence: float,
        audio: bool,
        logging_enabled: bool,
        save_images: bool
    ) -> str:
        """
        Aplica as configurações.
        
        Args:
            confidence: Limiar de confiança
            audio: Habilitar áudio
            logging_enabled: Habilitar logging
            save_images: Salvar imagens
            
        Returns:
            str: Mensagem de status
        """
        try:
            self.config.confidence_threshold = confidence
            self.config.audio_enabled = audio
            self.config.logging_enabled = logging_enabled
            self.config.save_images = save_images
            
            self.speaker.set_enabled(audio)
            self.logger.set_enabled(logging_enabled)
            
            return "✅ Configurações aplicadas com sucesso!"
            
        except Exception as e:
            self.logger_interface.error(f"Erro ao aplicar configurações: {e}")
            return f"❌ Erro ao aplicar configurações: {e}"
    
    def _get_statistics(self) -> str:
        """
        Obtém estatísticas das detecções.
        
        Returns:
            str: Estatísticas formatadas
        """
        stats = self.logger.get_statistics()
        
        if stats is None:
            return "Nenhuma detecção registrada ainda."
        
        text = f"""
Total de Detecções: {stats['total_detections']}
Objetos Únicos: {stats['unique_objects']}
Confiança Média: {stats['avg_confidence']:.2f}%

Objetos Mais Comuns:
"""
        for obj, count in stats['object_counts'].items():
            text += f"  - {obj}: {count}\n"
        
        text += f"""
Período:
  - Início: {stats['date_range']['start']}
  - Fim: {stats['date_range']['end']}
"""
        
        return text
    
    def launch(self):
        """Inicia a interface Gradio."""
        self.logger_interface.info("Iniciando interface Gradio...")
        self.interface.launch(
            share=self.config.gradio_share,
            server_name=self.config.gradio_server_name,
            server_port=self.config.gradio_port
        )
    
    def __repr__(self) -> str:
        """Representação string da interface."""
        return f"GradioInterface(port={self.config.gradio_port}, share={self.config.gradio_share})"
