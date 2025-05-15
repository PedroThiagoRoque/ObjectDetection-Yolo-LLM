import os
import time
import base64
from typing import Generator
from plyer import notification
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

"""
Gradio Streaming – Detector de Objetos em Tempo Real + Resumo GPT‑4o (Visão)
===========================================================================
Mostra o vídeo quadro a quadro com boxes do YOLO e gera a cada
"summary_interval" segundos, um resumo curto da cena usando o endpoint
(`input_image`) da OpenAI.

Requisitos:
    pip install ultralytics opencv-python python-dotenv "openai>=1.14.0" gradio numpy

Arquivo ".env" com:
    OPENAI_API_KEY="taldachave"
"""

# ---------- Utilidades LLM ---------- #

def summarize_frame(frame_bgr: np.ndarray, client: OpenAI) -> str:
    # 1) redimensiona para economizar tokens
    h, w = frame_bgr.shape[:2]
    scale = 512 / min(h, w)
    if scale < 1:  # só downscale
        frame_bgr = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)))

    # 2) BGR→RGB e JPEG
    frame_rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ok, jpeg    = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return "[Falha ao codificar frame]"

    b64 = base64.b64encode(jpeg).decode()

    # 3) Chat multimodal
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=60,
        messages=[{
            "role": "user",
            "content": [
                { "type": "text",
                  "text": "Resuma o que há na imagem em até 2 frases, em português." },
                { "type": "image_url",
                  "image_url": { "url": f"data:image/jpeg;base64,{b64}",
                                 "detail": "low" } }
            ]
        }]
    )
    return chat.choices[0].message.content.strip()


# ---------- Streaming Generator ---------- #

def stream_video(video_path: str,
                 model_path: str,
                 tags_str: str,
                 summary_interval: int = 5):
    """
    Lê vídeo, faz detecção YOLO, descreve o frame via GPT-4o-mini
    e emite notificação desktop se tags de interesse aparecerem
    (checa apenas a cada summary_interval p/ evitar spam).
    """
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None:
        raise RuntimeError("Defina OPENAI_API_KEY em um .env ou variável de ambiente")

    tags = [t.strip().lower() for t in tags_str.split(",") if t.strip()]
    client = OpenAI()
    model  = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo enviado.")

    FONT              = cv2.FONT_HERSHEY_SIMPLEX
    last_summary_time = 0.0
    summary_text      = "Início…"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- YOLO ----------
        results   = model(frame, verbose=False)
        boxes     = results[0].boxes
        yolo_lbls = [model.model.names[int(c)] for c in boxes.cls]
        annotated = results[0].plot()           # BGR

        # ---------- GPT-4o ----------
        now = time.time()
        if now - last_summary_time >= summary_interval:
            try:
                summary_text = summarize_frame(frame, client)
            except Exception as exc:
                summary_text = f"[Erro LLM] {exc}"

            last_summary_time = now  # zera o cronômetro

            # ---------- Notificação (1x por intervalo) ----------
            if tags:
                resumo_lc = summary_text.lower()
                acertos   = {tag for tag in tags
                             if tag in resumo_lc or tag in yolo_lbls}
                if acertos:
                    notification.notify(
                        title="Tag detectada",
                        message=f"{', '.join(sorted(acertos))}\n\n{summary_text}",
                        timeout=6  # segundos
                    )

        # ---------- Streaming ----------
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        yield annotated_rgb, summary_text

    cap.release()



# ---------- Interface Gradio ---------- #

def build_interface():
    with gr.Blocks(title="Detector YOLO + GPT-4o – Streaming") as demo:
        gr.Markdown("## Detecção em Tempo Real + Resumo Multimodal GPT-4o-mini")
        with gr.Row():
            video_input    = gr.Video(label="Vídeo de entrada")
            model_choice   = gr.Textbox(value="yolov8s.pt", label="Modelo YOLO (.pt)")
            tags_box       = gr.Textbox(label="Tags de interesse (vírgulas)", placeholder="pessoa, gato, carro")
            interval_slider= gr.Slider(1, 30, value=10, step=1, label="Intervalo (s) entre descrições")

        run_button = gr.Button("Processar em tempo real")

        with gr.Row():
            frame_output   = gr.Image(label="Frame anotado", type="numpy")
            summary_output = gr.Textbox(label="Descrição atual", max_lines=4)

        run_button.click(
            fn=stream_video,
            inputs=[video_input, model_choice, tags_box, interval_slider],
            outputs=[frame_output, summary_output],
            show_progress=True,
            queue=True,
        )
    return demo


if __name__ == "__main__":
    ui = build_interface()
    ui.queue()  # suporte a generators
    ui.launch()
