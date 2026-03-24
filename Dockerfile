# Dockerfile para Sistema de Detecção de Objetos

FROM python:3.11-slim

# Metadados
LABEL maintainer="ronbragaglia@gmail.com"
LABEL description="Sistema de Detecção de Objetos em Tempo Real com YOLOv5 e Áudio"
LABEL version="1.0.0"

# Definir variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Criar diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    # Dependências para OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Dependências para áudio
    alsa-utils \
    # Ferramentas úteis
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de dependências
COPY requirements.txt pyproject.toml ./

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código do projeto
COPY src/ ./src/
COPY main.py ./
COPY .env.example .env

# Criar diretórios necessários
RUN mkdir -p models data logs output detections_images

# Criar usuário não-root para segurança
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Trocar para usuário não-root
USER appuser

# Expor porta do Gradio
EXPOSE 7860

# Comando de execução
CMD ["python", "main.py"]
