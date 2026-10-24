import os
import sys

# Compatibility shim: In Hugging Face Spaces Gradio containers with websockets < 13,
# patch ClientConnection to ensure marimo ASGI server boots seamlessly.
try:
    import websockets
    if not hasattr(websockets, "ClientConnection"):
        try:
            from websockets.legacy.client import WebSocketClientProtocol
            websockets.ClientConnection = WebSocketClientProtocol
        except ImportError:
            try:
                from websockets.client import WebSocketClientProtocol
                websockets.ClientConnection = WebSocketClientProtocol
            except ImportError:
                pass
except Exception:
    pass

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import marimo
import uvicorn

# 1. Build Marimo ASGI applications (Main Dashboard and Widget Showcase)
builder = (
    marimo.create_asgi_app()
    .with_app(path="", root="app.py")
    .with_app(path="/widgets", root="marimo_widgets_showcase.py")
)
marimo_asgi = builder.build()

# 2. Top-level FastAPI application
app = FastAPI(title="CardioPulse Clinical Intelligence")

# 3. Mount Marimo onto /marimo
app.mount("/marimo", marimo_asgi)

# 4. Gradio interface providing Hugging Face healthcheck endpoints & launchpad
with gr.Blocks(
    title="CardioPulse | Clinical Readmission Analytics",
    theme=gr.themes.Soft(primary_hue="teal"),
    css="footer {visibility: hidden !important;}"
) as demo:
    gr.Markdown(
        """
        # 🏥 CardioPulse: Hospital Readmission Analytics & Clinical Risk Intelligence
        
        This platform runs on **Marimo**, an interactive reactive Python notebook with an in-memory **DuckDB** SQL engine and **Scikit-Learn** predictive triage.
        
        *Redirecting to the reactive application...*
        """
    )
    with gr.Row():
        gr.Button("🚀 Open Clinical Dashboard", link="/marimo")
        gr.Button("🧩 Open Marimo Widget Showcase", link="/marimo/widgets")
        
    gr.HTML('<meta http-equiv="refresh" content="0; url=/marimo" />')

# 5. Mount Gradio so Hugging Face Spaces runtime supervisor detects a healthy Gradio app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
