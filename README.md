📌 Detecção de Objetos em Tempo Real com YOLOv5 e Áudio

🔍 Projeto de detecção de objetos usando YOLOv5, OpenCV e Gradio com saída em áudio
📡 Rodando localmente e compartilhando via Gradio
🎤 Alertas de voz sobre objetos detectados

 Tecnologias Utilizadas
Python 3.13
YOLOv5s (Ultralytics)
OpenCV (para captura e exibição de vídeo)
gTTS (Google Text-to-Speech para avisos em áudio)
Gradio (para criar interface de usuário interativa)
Pandas (para registro e análise de detecções)
PyTorch (para rodar a inferência do YOLOv5)

📌 Como Funciona
Captura da webcam usando OpenCV.
Processamento com YOLOv5s para detectar objetos na imagem em tempo real.
Caixas delimitadoras (Bounding Boxes) são desenhadas na tela para cada objeto detectado.
Geração de áudio com gTTS para anunciar o objeto detectado.
Registro em logs: Cada detecção é salva em um arquivo .txt e analisada para gerar um relatório em .csv.
Interface com Gradio permite iniciar e controlar o sistema via um link gerado automaticamente.

📌 Precisão do Modelo
O YOLOv5s tem alta precisão para objetos comuns. Durante os testes, foram obtidos os seguintes valores médios de confiança:

📌 Objeto	📊 Precisão Média (%)
🧑‍🤝‍🧑 Pessoa	85-90%
📱 Celular	70-80%
📺 TV	65-80%
🌱 Planta	30-40%
💻 Laptop	50-80%
A precisão varia dependendo da iluminação, distância e qualidade da câmera.

📌 Resultados
📸 Exemplo de detecção:
✅ O modelo conseguiu identificar corretamente pessoas, celulares, televisões, plantas, laptops e outros objetos em tempo real.
✅ Saída de áudio funcionando, avisando os objetos detectados.
✅ Logs e relatórios criados automaticamente para análise posterior.
✅ Interface Gradio acessível via link público, permitindo o uso remoto sem precisar instalar nada.

![image](https://github.com/user-attachments/assets/30445a40-1a20-42b0-bf80-b35cccbbbe3a)

📌 Como Usar
1️⃣ Instalar Dependências
bash
Copiar
Editar
pip install torch torchvision torchaudio ultralytics opencv-python pandas gradio gtts

2️⃣ Rodar o Código
bash
Copiar
Editar
python detecao_objetos.py
A interface será aberta no navegador e a detecção será iniciada.

📌 Contribuições
💡 Caso tenha sugestões ou melhorias, fique à vontade para abrir uma issue ou um pull request no repositório!

📢 Autor: Rone Bragaglia
📌 Repositório: github.com/Ronbragaglia/Deteccao-Objetos



