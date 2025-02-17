import torch
import cv2
import threading
import os
import time
import subprocess
import pandas as pd
import gradio as gr
from gtts import gTTS
from datetime import datetime
from ultralytics import YOLO


model = YOLO("yolov5su.pt")

detection_log = "detections_log.csv"
image_save_path = "detections_images"
os.makedirs(image_save_path, exist_ok=True) 

def speak(text):
    """Converte texto em fala usando gTTS e evita erro de permissão."""
    temp_audio_path = "audio_temp.mp3"
    tts = gTTS(text=text, lang='pt')
    tts.save(temp_audio_path)
    subprocess.call(["start", "", temp_audio_path], shell=True)  

def log_detection(detections):
    """Salva logs das detecções em CSV e captura imagens."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detected_objects = []
    
    for label, confidence, frame in detections:
        detected_objects.append([timestamp, label, f"{confidence:.2f}%"])
        img_filename = f"{image_save_path}/{timestamp.replace(':', '-')}_{label}.jpg"
        cv2.imwrite(img_filename, frame)
    
    df = pd.DataFrame(detected_objects, columns=["Timestamp", "Object", "Confidence"])
    df.to_csv(detection_log, mode='a', header=not os.path.exists(detection_log), index=False)

def object_detection_webcam():
    """Executa a detecção de objetos na webcam."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("🚨 Erro: Não foi possível acessar a webcam!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                label = model.names[int(box.cls[0].item())]
                
                if confidence > 0.3:  # Evita detecções de baixa confiança
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.1%})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detection_text = f"Detectado: {label} com {confidence:.1%} de confiança."
                    print(detection_text)
                    detections.append((label, confidence * 100, frame))
                    speak(detection_text)
        
        if detections:
            log_detection(detections)
        
        cv2.imshow("Detecção de Objetos - YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def start_object_detection_thread():
    """Inicia a detecção de objetos em uma thread para evitar travamento."""
    detection_thread = threading.Thread(target=object_detection_webcam)
    detection_thread.daemon = True
    detection_thread.start()

def detect_objects_from_image(image):
    """Permite enviar imagens e detectar objetos via interface Gradio."""
    results = model(image)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label = model.names[int(box.cls[0].item())]
            
            if confidence > 0.3:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label} ({confidence:.1%})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def gradio_interface():
    """Interface gráfica interativa."""
    def start_detection():
        start_object_detection_thread()
        return "🔍 Detecção iniciada! Acesse a câmera e veja os resultados."
    
    interface = gr.Interface(
        fn=detect_objects_from_image,
        inputs=gr.Image(type="numpy"),
        outputs=gr.Image(type="numpy"),
        title="🎯 Detecção de Objetos com YOLOv5 e Áudio",
        description="Envie uma imagem para detectar objetos ou clique no botão para usar a webcam.",
        live=True
    )
    interface.launch(share=True)  

if __name__ == "__main__":
    print("📷 Iniciando detecção de objetos em tempo real com áudio e logs...")
    gradio_interface()


        
