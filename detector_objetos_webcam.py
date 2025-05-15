import os
import time
from typing import Generator, Tuple

import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

"""
Gradio Streaming – Detector de Objetos em Tempo Real + Resumo GPT‑4o
=================================================================
Agora o vídeo é **mostrado em tempo real** (frame‑a‑frame) enquanto é processado.
Cada `yield` da função `stream_video` envia o frame anotado (JPEG em bytes) e o
texto de resumo atual ao front‑end Gradio.

Diferenças principais:
----------------------
• **Não** escrevemos um arquivo de saída; em vez disso, emitimos frames em vivo.
• A UI muda o componente de saída para `gr.Image`, pois é o único que Gradio
  consegue atualizar em tempo real (um `<video>` não pode ser atualizado por
  generator no momento¹).
• O processamento começa assim que você clica em **Processar**; os frames vão
  aparecendo enquanto o loop avança.

¹ Se você quer streaming em `<video>` verdadeiro (HLS / WebRTC), precisa de
  servidor de mídia; fora do escopo desta demo.

Instalação:
    pip install ultralytics opencv-python python-dotenv "openai>=1.14.0" gradio
    python detector_objetos_gradio_stream.py

Coloque no `.env`:
    OPENAI_API_KEY="sua_chave_aqui"
"""

# ---------- Resumo + LLM ---------- #

def summarize_labels(client: OpenAI, labels: list[str]) -> str:
    """Gera um resumo curto em PT‑BR via Responses API."""
    if not labels:
        return "Nenhum objeto detectado."

    messages = [
        {
            "role": "developer",
            "content": (
                "Você é um narrador de cenas ao vivo: descreva a situação para um público "
                "leigo de forma concisa (≤ 2 frases), em português do Brasil, sem repetir "
                "substantivos e priorizando humanos, depois veículos."
            ),
        },
        {
            "role": "user",
            "content": f"Detectei os seguintes objetos: {', '.join(labels)}",
        },
    ]

    response = client.responses.create(model="gpt-4o-mini", input=messages)
    return response.output_text.strip()


# ---------- Streaming Generator ---------- #

def stream_video(video_path: str, model_path: str, summary_interval: int = 5):
    """Yield contínuo de (frame numpy RGB, resumo str) para Gradio."""
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None:
        raise RuntimeError("Defina OPENAI_API_KEY em um .env ou variável de ambiente")

    client = OpenAI()
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo enviado.")

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    last_summary_time = 0.0
    summary_text = "Início…"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO
        results = model(frame, verbose=False)
        annotated = results[0].plot()  # BGR
        labels = [model.model.names[int(cls)] for cls in results[0].boxes.cls]

        now = time.time()
        if now - last_summary_time >= summary_interval:
            try:
                summary_text = summarize_labels(client, labels)
            except Exception as exc:
                summary_text = f"[Erro LLM] {exc}"
            last_summary_time = now

        cv2.putText(annotated, summary_text, (10, 30), FONT, 0.7, (0, 255, 0), 2)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        # Gradio aceita numpy array (H,W,3) em RGB
        yield annotated_rgb, summary_text

    cap.release()

# ---------- Interface Gradio ---------- #

def build_interface():
    with gr.Blocks(title="Detector de Objetos (YOLO + GPT-4o) – Streaming") as demo:
        gr.Markdown("## Detecção em Tempo Real + Resumo GPT‑4o (Streaming)")
        gr.Markdown(
            "Envie um vídeo; o processamento ocorrerá quadro a quadro e você verá o resultado em tempo real."
        )

        with gr.Row():
            video_input = gr.Video(label="Vídeo de entrada")
            model_choice = gr.Textbox(value="yolov8n.pt", label="Modelo YOLO (.pt)")
            interval_slider = gr.Slider(1, 10, value=5, step=1, label="Intervalo (s) para resumos")

        run_button = gr.Button("Processar em tempo real")

        with gr.Row():
            frame_output = gr.Image(label="Frame anotado (stream)", type="numpy")
            summary_output = gr.Textbox(label="Resumo atual", max_lines=4)

            run_button.click(
            fn=stream_video,
            inputs=[video_input, model_choice, interval_slider],
            outputs=[frame_output, summary_output],
            show_progress=True,
            queue=True
        )

    return demo


if __name__ == "__main__":
    ui = build_interface()
    ui.queue()  # garante suporte a generators
    ui.launch()
