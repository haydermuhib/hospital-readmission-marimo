import os
import marimo
import uvicorn

# Build production ASGI app serving both the main dashboard and the widget reference
builder = (
    marimo.create_asgi_app()
    .with_app(path="", root="app.py")
    .with_app(path="/widgets", root="marimo_widgets_showcase.py")
)
app = builder.build()

if __name__ == "__main__":
    # Hugging Face Spaces automatically injects the PORT environment variable (default 7860)
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
