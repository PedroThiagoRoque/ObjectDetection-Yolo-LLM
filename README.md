# 📸🦾 YOLO + GPT‑4o Live Vision
---

## O que este projeto faz

* Captura **vídeo ao vivo** (webcam ou arquivo) e roda **YOLOv8** para desenhar *bounding‑boxes* dos objetos.
* A cada *N* segundos, manda o **frame** pro **GPT‑4o** multimodal, que devolve um resumo curtinho em português.
* Exibe tudo **em tempo real** numa interface **Gradio**: frame anotado de um lado, descrição do outro.

![demo-gif](docs/demo.gif)

---

## 🛠️ Tecnologias

| Peça                   | Função                                           |
| ---------------------- | ------------------------------------------------ |
| **Python 3.11+**       | Base de tudo                                     |
| **OpenCV**             | Para ler vídeo e desenhar coisas                 |
| **Ultralytics YOLOv8** | Detecção local de objetos                        |
| **OpenAI SDK v1**      | Chat multimodal GPT‑4o                           |
| **Gradio**             | UI web rapidinha                                 |
| **python‑dotenv**      | Chave `OPENAI_API_KEY=`                          |

---

### 🔑 Configurar API Key

Crie um arquivo `.env` na raiz (mesmo nível do script) com:

```
OPENAI_API_KEY="suachave"
```

---

## 🚀 Como rodar

> Tudo testado no Windows 10. Deve funcionar em qualquer lugar onde o Python roda.

```bash
# 1. clone o repo (ou baixe o zip)

# 2. crie o ambiente virtual — opcional
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows PowerShell

# 3. instale as dependências
py -3.12 -m pip install -r requirements.txt

# 4. Rodar
py -3.12 detector_objetos_webcam.py # abre no browser http://127.0.0.1:7860

```

---

### 🌐 UI Gradio

1. Faça upload de um vídeo **(ou coloque para webcam)**
2. Clique em **Processar em tempo real**
3. Observe as etiquetas do yolo e os resumos de frames

---

## 🏎️ Dicas de performance

* GPU NVIDIA? Instale PyTorch com CUDA (veja docs do Ultralytics).
* Mude `detail":"low"` → `"high"` só quando precisar de descrições super específicas (vai custar +tokens).
* Redimensiona os frames pra 512 px – bom o suficiente na maioria dos cenários.

---