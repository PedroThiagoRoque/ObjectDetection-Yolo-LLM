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

