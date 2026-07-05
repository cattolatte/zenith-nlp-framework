"""FastAPI generation service — Zenith's product surface.

Design Principles
-----------------
A thin HTTP layer over a trained checkpoint: load it once, expose text generation
(blocking) and token streaming (Server-Sent Events) over the same decoding
controls the ``Generator`` offers. FastAPI/uvicorn/pydantic are imported lazily
inside ``create_app`` so ``import zenith.serving`` needs no serving dependencies.

Note: this module intentionally does *not* use ``from __future__ import
annotations`` — FastAPI resolves the endpoints' parameter annotations at
definition time, and the request/response models are defined inside
``create_app``, so their annotations must bind to the live classes (not strings).
"""

import json
from functools import lru_cache
from typing import Any

from .. import __version__

__all__ = ["create_app", "serve"]


def create_app(model_path: str) -> Any:
    """Build a FastAPI app serving the checkpoint at ``model_path``."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    from ..checkpoint import load_pretrained

    app = FastAPI(title="Zenith Serving", version=__version__)

    class GenerateRequest(BaseModel):
        prompt: str = ""
        max_new_tokens: int = 200
        temperature: float = 0.8
        top_k: int | None = None
        top_p: float | None = None
        repetition_penalty: float = 1.0
        num_beams: int | None = None

    class GenerateResponse(BaseModel):
        prompt: str
        completion: str

    @lru_cache(maxsize=1)
    def _generator() -> Any:
        return load_pretrained(model_path)

    @app.get("/health")
    def health() -> dict:
        try:
            _generator()
        except Exception as exc:  # pragma: no cover - surfaced to the client
            raise HTTPException(status_code=503, detail=f"model not loaded: {exc}") from exc
        return {"status": "ok", "model": model_path}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        completion = _generator().generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            num_beams=request.num_beams,
        )
        return GenerateResponse(prompt=request.prompt, completion=completion)

    @app.post("/generate/stream")
    def generate_stream(request: GenerateRequest) -> StreamingResponse:
        generator = _generator()

        def event_stream():
            for chunk in generator.stream(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def serve(model_path: str, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch the generation service with uvicorn."""
    import uvicorn

    uvicorn.run(create_app(model_path), host=host, port=port)
