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
MODEL_PATH = "yolov8s.pt"       # ou yolov8n.pt para +performance
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

# ---------- Funções Auxiliares ---------- #

def summarize_labels(labels: list[str]) -> str:
    """Envia as labels para o GPT‑4o e devolve um resumo curto em PT‑BR."""
    if not labels:
        return "Nenhum objeto detectado."

    prompt = (
        "Resuma brevemente, em português, o que está na cena, "
        f"dado que detectei os seguintes objetos: {', '.join(labels)}."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # use o modelo que estiver disponível
            messages=[
                {"role": "system", "content": "Você é um narrador de cenas ao vivo: descreva a situação para um público leigo de forma concisa (< 4 frases), em português do Brasil. Mencione quantidades, cores e movimentos perceptíveis; evite julgamento estético. Se nenhum objeto relevante, devolva exatamente a string ‘SEM_OBJETOS’. Caso contrário, siga as instruções anteriores."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=350,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"[Erro LLM] {exc}"



