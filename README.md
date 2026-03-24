# 🎯 Sistema de Detecção de Objetos em Tempo Real com YOLOv5 e Áudio

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Sistema completo de detecção de objetos em tempo real utilizando YOLOv5, OpenCV e Gradio com saída em áudio. O sistema oferece uma interface de usuário intuitiva, logging automático de detecções e suporte a múltiplas plataformas.

![Detecção de Objetos](https://github.com/user-attachments/assets/30445a40-1a20-42b0-bf80-b35cccbbbe3a)

## 📋 Índice

- [Características](#características)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Uso](#uso)
- [Configuração](#configuração)
- [Exemplos](#exemplos)
- [API](#api)
- [Desenvolvimento](#desenvolvimento)
- [Docker](#docker)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

## ✨ Características

- 🎯 **Detecção em Tempo Real**: Detecção de objetos usando YOLOv5 com alta precisão
- 🖥️ **Interface Gráfica**: Interface interativa com Gradio
- 🎤 **Síntese de Voz**: Anúncios de áudio para objetos detectados
- 📊 **Logging Automático**: Registro de detecções em CSV
- 💾 **Salvamento de Imagens**: Armazenamento automático de imagens detectadas
- ⚙️ **Configuração Flexível**: Personalização via arquivo .env
- 🐳 **Suporte a Docker**: Containerização fácil com Docker e Docker Compose
- 🧪 **Testes Automatizados**: Suíte completa de testes
- 📈 **Estatísticas**: Análise detalhada de detecções
- 🌐 **Multi-plataforma**: Funciona em Windows, Linux e macOS

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**: Linguagem principal
- **YOLOv5 (Ultralytics)**: Modelo de detecção de objetos
- **OpenCV**: Processamento de imagens e vídeo
- **Gradio**: Interface de usuário interativa
- **gTTS**: Google Text-to-Speech para síntese de voz
- **PyTorch**: Framework de deep learning
- **Pandas**: Manipulação de dados e logging

## 📥 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Webcam (para detecção em tempo real)

### Instalação via pip

1. Clone o repositório:

```bash
git clone https://github.com/Ronbragaglia/DeteccaoObjetos.git
cd DeteccaoObjetos
```

2. Crie um ambiente virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:

```bash
cp .env.example .env
# Edite o arquivo .env conforme necessário
```

### Instalação via Docker

1. Construa a imagem Docker:

```bash
docker build -t deteccao-objetos .
```

2. Execute o container:

```bash
docker run -p 7860:7860 deteccao-objetos
```

### Instalação via Docker Compose

```bash
docker-compose up -d
```

## 🚀 Uso

### Iniciar o Sistema

```bash
python main.py
```

O sistema iniciará automaticamente e abrirá uma interface no navegador.

### Usando a Interface Gráfica

1. **Aba de Detecção em Imagens**:
   - Carregue uma imagem
   - Clique em "Detectar Objetos"
   - Visualize os resultados

2. **Aba de Detecção em Tempo Real**:
   - Clique em "Iniciar Detecção"
   - A webcam será ativada
   - Pressione 'q' na janela OpenCV para parar

3. **Aba de Configurações**:
   - Ajuste o limiar de confiança
   - Habilite/desabilite áudio
   - Configure logging e salvamento de imagens

4. **Aba de Estatísticas**:
   - Visualize estatísticas de detecção
   - Exporte dados em diferentes formatos

## ⚙️ Configuração

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# Configurações do Modelo
MODEL_NAME=yolov5su.pt
CONFIDENCE_THRESHOLD=0.3
MAX_DETECTIONS=100
DEVICE=cpu

# Configurações de Áudio
AUDIO_ENABLED=true
AUDIO_LANGUAGE=pt

# Configurações de Logging
LOGGING_ENABLED=true
LOG_FILE=detections_log.csv
LOG_LEVEL=INFO

# Configurações de Imagens
SAVE_IMAGES=true
IMAGE_SAVE_PATH=detections_images
IMAGE_FORMAT=jpg
IMAGE_QUALITY=95

# Configurações de Webcam
WEBCAM_INDEX=0
WEBCAM_WIDTH=640
WEBCAM_HEIGHT=480
WEBCAM_FPS=30

# Configurações do Gradio
GRADIO_SHARE=true
GRADIO_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0
```

### Filtro de Classes

Você pode filtrar classes específicas para detecção:

```python
from src.config.config import Config

config = Config(
    allowed_classes=["person", "car", "dog"]
)
```

Ou excluir classes específicas:

```python
config = Config(
    excluded_classes=["person", "cell phone"]
)
```

## 📚 Exemplos

### Exemplo Básico

```python
from src.config.config import Config
from src.detection.detector import ObjectDetector
from src.audio.speaker import Speaker
from src.logging.logger import DetectionLogger

# Criar configuração
config = Config()

# Inicializar componentes
detector = ObjectDetector(config)
speaker = Speaker(config)
logger = DetectionLogger(config)

# Detectar objetos
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detections, annotated_frame = detector.detect(frame)
    
    for detection in detections:
        speaker.speak_detection(detection.label, detection.confidence)
    
    logger.log_detections(detections, frame)
    cv2.imshow("Detecção", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### Exemplo Avançado

Consulte os arquivos em [`examples/`](examples/) para exemplos mais detalhados:

- [`basic_usage.py`](examples/basic_usage.py): Exemplo básico de uso
- [`advanced_usage.py`](examples/advanced_usage.py): Exemplo avançado com filtros e personalização

Execute os exemplos:

```bash
python examples/basic_usage.py
python examples/advanced_usage.py
```

## 📖 API

### Config

```python
from src.config.config import Config

# Criar configuração padrão
config = Config()

# Criar configuração personalizada
config = Config(
    model_name="yolov8n.pt",
    confidence_threshold=0.7,
    audio_enabled=False
)

# Carregar do ambiente
config = Config.from_env()

# Carregar de dicionário
config = Config.from_dict({
    "model_name": "yolov8n.pt",
    "confidence_threshold": 0.7
})
```

### ObjectDetector

```python
from src.detection.detector import ObjectDetector

detector = ObjectDetector(config)

# Detectar objetos
detections, annotated_frame = detector.detect(frame)

# Obter informações do modelo
info = detector.get_model_info()

# Obter nomes das classes
classes = detector.get_class_names()
```

### Speaker

```python
from src.audio.speaker import Speaker

speaker = Speaker(config)

# Falar texto
speaker.speak("Olá, mundo!")

# Falar detecção
speaker.speak_detection("person", 0.85)

# Habilitar/desabilitar áudio
speaker.set_enabled(True)
speaker.set_enabled(False)
```

### DetectionLogger

```python
from src.logging.logger import DetectionLogger

logger = DetectionLogger(config)

# Logar detecções
logger.log_detections(detections, frame)

# Obter estatísticas
stats = logger.get_statistics()

# Exportar log
logger.export_log("output.csv", format="csv")
logger.export_log("output.json", format="json")
```

## 🧪 Desenvolvimento

### Executar Testes

```bash
# Executar todos os testes
pytest

# Executar com cobertura
pytest --cov=src --cov-report=html

# Executar testes específicos
pytest tests/test_config.py

# Executar apenas testes unitários
pytest -m unit
```

### Formatação de Código

```bash
# Formatar código com Black
black src/ tests/

# Verificar estilo com Flake8
flake8 src/ tests/

# Verificar tipos com MyPy
mypy src/
```

### Estrutura do Projeto

```
DeteccaoObjetos/
├── src/                    # Código fonte
│   ├── config/            # Configurações
│   ├── detection/         # Detecção de objetos
│   ├── audio/             # Síntese de voz
│   ├── logging/           # Logging de detecções
│   └── interface/         # Interface Gradio
├── tests/                 # Testes
├── examples/              # Exemplos de uso
├── docs/                  # Documentação
├── data/                  # Dados
├── logs/                  # Logs
├── models/                # Modelos
├── output/                # Saídas
├── .github/               # GitHub Actions
├── main.py               # Ponto de entrada
├── requirements.txt      # Dependências
├── pyproject.toml        # Configuração do projeto
├── Dockerfile            # Docker
├── docker-compose.yml    # Docker Compose
└── README.md             # Este arquivo
```

## 🐳 Docker

### Construir Imagem

```bash
docker build -t deteccao-objetos .
```

### Executar Container

```bash
docker run -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  deteccao-objetos
```

### Docker Compose

```bash
# Iniciar todos os serviços
docker-compose up -d

# Parar todos os serviços
docker-compose down

# Visualizar logs
docker-compose logs -f
```

## 📊 Precisão do Modelo

O YOLOv5s tem alta precisão para objetos comuns. Durante os testes, foram obtidos os seguintes valores médios de confiança:

| Objeto | Precisão Média |
|---------|----------------|
| 🧑‍🤝‍🧑 Pessoa | 85-90% |
| 📱 Celular | 70-80% |
| 📺 TV | 65-80% |
| 💻 Laptop | 50-80% |
| 🌱 Planta | 30-40% |

A precisão varia dependendo da iluminação, distância e qualidade da câmera.

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, leia o [`CONTRIBUTING.md`](CONTRIBUTING.md) para detalhes sobre como contribuir.

### Como Contribuir

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Add: Minha Feature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [`LICENSE`](LICENSE) para detalhes.

## 👤 Autor

**Rone Bragaglia**

- GitHub: [@Ronbragaglia](https://github.com/Ronbragaglia)
- Email: ronbragaglia@gmail.com

## 🙏 Agradecimentos

- [Ultralytics](https://github.com/ultralytics/ultralytics) pelo YOLOv5
- [Gradio](https://gradio.app/) pela interface de usuário
- [OpenCV](https://opencv.org/) pelo processamento de imagens
- [Google](https://cloud.google.com/text-to-speech) pelo gTTS

## 📞 Suporte

Se você tiver alguma dúvida ou problema:

- Abra uma [issue](https://github.com/Ronbragaglia/DeteccaoObjetos/issues)
- Envie um email para ronbragaglia@gmail.com

## 📝 Changelog

Veja o [`CHANGELOG.md`](CHANGELOG.md) para informações sobre mudanças recentes.

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela! ⭐**
