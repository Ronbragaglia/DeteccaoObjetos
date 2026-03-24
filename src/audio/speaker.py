"""
Módulo de síntese de voz.

Este módulo fornece uma classe para conversão de texto em fala
usando a biblioteca gTTS (Google Text-to-Speech).
"""

import os
import subprocess
import tempfile
import logging
from typing import Optional
from pathlib import Path

try:
    from gtts import gTTS
except ImportError:
    raise ImportError(
        "gTTS não está instalado. "
        "Instale com: pip install gtts"
    )

from ..config.config import Config


class Speaker:
    """
    Classe para síntese de voz usando gTTS.
    
    Esta classe encapsula toda a lógica de conversão de texto em fala,
    incluindo geração de áudio e reprodução.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Inicializa o speaker.
        
        Args:
            config: Configuração do speaker. Se None, usa configuração padrão.
        """
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        self._temp_audio_path = None
    
    def speak(self, text: str, blocking: bool = False) -> bool:
        """
        Converte texto em fala e reproduz.
        
        Args:
            text: Texto a ser convertido em fala
            blocking: Se True, aguarda a reprodução terminar
            
        Returns:
            bool: True se a fala foi reproduzida com sucesso, False caso contrário
        """
        if not self.config.audio_enabled:
            self.logger.debug("Áudio desabilitado, ignorando fala")
            return False
        
        if not text or not text.strip():
            self.logger.warning("Texto vazio, ignorando fala")
            return False
        
        try:
            # Gerar áudio
            audio_path = self._generate_audio(text)
            
            if audio_path is None:
                return False
            
            # Reproduzir áudio
            success = self._play_audio(audio_path, blocking=blocking)
            
            # Limpar arquivo temporário
            self._cleanup_audio(audio_path)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erro ao reproduzir fala: {e}")
            return False
    
    def _generate_audio(self, text: str) -> Optional[str]:
        """
        Gera um arquivo de áudio a partir do texto.
        
        Args:
            text: Texto a ser convertido
            
        Returns:
            Optional[str]: Caminho do arquivo de áudio gerado, ou None em caso de erro
        """
        try:
            # Criar arquivo temporário
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # Gerar áudio usando gTTS
            tts = gTTS(
                text=text,
                lang=self.config.audio_language,
                slow=self.config.audio_slow
            )
            tts.save(temp_path)
            
            self.logger.debug(f"Áudio gerado: {temp_path}")
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar áudio: {e}")
            return None
    
    def _play_audio(self, audio_path: str, blocking: bool = False) -> bool:
        """
        Reproduz um arquivo de áudio.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            blocking: Se True, aguarda a reprodução terminar
            
        Returns:
            bool: True se a reprodução foi bem-sucedida, False caso contrário
        """
        try:
            # Detectar sistema operacional
            if os.name == 'nt':  # Windows
                return self._play_audio_windows(audio_path, blocking)
            elif os.name == 'posix':  # Linux/Mac
                return self._play_audio_posix(audio_path, blocking)
            else:
                self.logger.error(f"Sistema operacional não suportado: {os.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao reproduzir áudio: {e}")
            return False
    
    def _play_audio_windows(self, audio_path: str, blocking: bool = False) -> bool:
        """
        Reproduz áudio no Windows.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            blocking: Se True, aguarda a reprodução terminar
            
        Returns:
            bool: True se a reprodução foi bem-sucedida
        """
        try:
            if blocking:
                # Reprodução bloqueante
                subprocess.call(
                    ["powershell", "-c", f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync()"],
                    shell=True
                )
            else:
                # Reprodução não-bloqueante
                subprocess.Popen(
                    ["powershell", "-c", f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync()"],
                    shell=True
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao reproduzir áudio no Windows: {e}")
            return False
    
    def _play_audio_posix(self, audio_path: str, blocking: bool = False) -> bool:
        """
        Reproduz áudio no Linux/Mac.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            blocking: Se True, aguarda a reprodução terminar
            
        Returns:
            bool: True se a reprodução foi bem-sucedida
        """
        try:
            # Tentar diferentes players
            players = [
                ["aplay", audio_path],  # Linux
                ["afplay", audio_path],  # Mac
                ["mpg123", audio_path],  # Linux
                ["ffplay", "-nodisp", "-autoexit", audio_path],  # Multi-plataforma
            ]
            
            for player in players:
                try:
                    if blocking:
                        subprocess.call(player)
                    else:
                        subprocess.Popen(player)
                    
                    return True
                    
                except FileNotFoundError:
                    continue
            
            self.logger.error("Nenhum player de áudio encontrado")
            return False
            
        except Exception as e:
            self.logger.error(f"Erro ao reproduzir áudio no POSIX: {e}")
            return False
    
    def _cleanup_audio(self, audio_path: str):
        """
        Remove o arquivo de áudio temporário.
        
        Args:
            audio_path: Caminho do arquivo de áudio
        """
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                self.logger.debug(f"Arquivo de áudio removido: {audio_path}")
        except Exception as e:
            self.logger.warning(f"Erro ao remover arquivo de áudio: {e}")
    
    def speak_detection(self, label: str, confidence: float) -> bool:
        """
        Fala uma detecção de objeto.
        
        Args:
            label: Rótulo do objeto detectado
            confidence: Confiança da detecção (0-1)
            
        Returns:
            bool: True se a fala foi bem-sucedida
        """
        text = f"Detectado: {label} com {confidence:.1%} de confiança."
        return self.speak(text)
    
    def is_enabled(self) -> bool:
        """
        Verifica se o áudio está habilitado.
        
        Returns:
            bool: True se o áudio está habilitado
        """
        return self.config.audio_enabled
    
    def set_enabled(self, enabled: bool):
        """
        Habilita ou desabilita o áudio.
        
        Args:
            enabled: True para habilitar, False para desabilitar
        """
        self.config.audio_enabled = enabled
        self.logger.info(f"Áudio {'habilitado' if enabled else 'desabilitado'}")
    
    def __repr__(self) -> str:
        """Representação string do speaker."""
        return f"Speaker(enabled={self.config.audio_enabled}, language={self.config.audio_language})"
