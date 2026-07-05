# Phase 5 — Serving & product surface

## Goal
Turn a trained checkpoint into something you can talk to: an HTTP service with
streaming, a CLI to launch it, and an interactive REPL.

## Scope (what ships this phase)
- `serving/` — `create_app(model_path)` (FastAPI) + `serve()`:
  - `GET /health`
  - `POST /generate` — full decoding controls (temperature/top-k/top-p/
    repetition_penalty/num_beams), returns the completion.
  - `POST /generate/stream` — **Server-Sent Events**, one JSON `{"text": ...}`
    per chunk, terminated by `[DONE]`.
- `Generator.stream()` — yields decoded text incrementally, buffering bytes so a
  partial multibyte character is never emitted; uses the KV-cache.
- CLI: `zenith serve -m <ckpt>` and an interactive `zenith chat -m <ckpt>` REPL
  that streams to the terminal.
- Packaging: `serving` extra (fastapi/uvicorn/pydantic); `httpx` in `dev` for the
  test client; CI installs the serving extra.
- Tests: health / generate / SSE stream (FastAPI `TestClient`); `Generator.stream`
  yields strings.

## Key decisions (ADR-0006)
- **Streaming is SSE with UTF-8-complete chunks.** SSE is the simplest broadly-
  compatible transport and needs no extra dependency (`StreamingResponse`); the
  byte-buffering keeps chunks valid for a byte-level tokenizer.
- **Serving is a thin layer over `Generator`.** No generation logic in the web
  layer; the checkpoint is loaded once (lazily, cached).
- FastAPI/uvicorn are lazy, optional extras — `import zenith.serving` stays cheap.

## Out of scope (later)
- Multi-model serving, token-usage metrics, batching, auth.

## Definition of done
`zenith serve` / `zenith chat` work; `/generate` and `/generate/stream` are tested;
gates green; tag `v0.5.0`.
