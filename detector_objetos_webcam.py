#Detector de Objetos em Tempo Real com YOLOv8 + GPT‑4o
#----------------------------------------------------
#- Captura vídeo da webcam.
#- Detecta objetos localmente com YOLOv8 (modelo nano para velocidade).
#- A cada X segundos, envia as classes detectadas ao LLM da OpenAI para receber um resumo em linguagem natural português.
#- Exibe o vídeo com bounding‑boxes e o resumo na tela.


import os
import time

import cv2 
from ultralytics import YOLO
import openai
from dotenv import load_dotenv

# ---------- Configurações ---------- #
MODEL_PATH = "yolov8n.pt"       # ou yolov8s.pt para +qualidade
SUMMARY_INTERVAL = 5            # segundos entre chamadas ao GPT‑4o
FONT = cv2.FONT_HERSHEY_SIMPLEX # fonte do OpenCV

# ---------- Inicialização ---------- #
load_dotenv()  # carrega variáveis do arquivo .env
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise RuntimeError("Defina OPENAI_API_KEY no arquivo .env")

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam")

last_summary_time = 0.0
summary_text = "Inicializando…"
