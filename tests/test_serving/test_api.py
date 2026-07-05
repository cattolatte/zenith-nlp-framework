"""Tests for the FastAPI generation service (via a randomly-initialised model)."""

from fastapi.testclient import TestClient

from zenith.checkpoint import save_checkpoint
from zenith.models import DecoderConfig, DecoderLM
from zenith.serving import create_app
from zenith.tokenizers import ByteTokenizer


def _client(tmp_path) -> TestClient:
    model = DecoderLM(
        DecoderConfig(vocab_size=259, block_size=32, embed_dim=32, num_layers=2,
                      num_heads=2, ff_dim=64, dropout=0.0)
    )
    path = tmp_path / "lm.pt"
    save_checkpoint(model, ByteTokenizer(), str(path))
    return TestClient(create_app(str(path)))


def test_health_ok(tmp_path):
    response = _client(tmp_path).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate_returns_completion(tmp_path):
    response = _client(tmp_path).post("/generate", json={"prompt": "hi", "max_new_tokens": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["prompt"] == "hi"
    assert isinstance(body["completion"], str)


def test_stream_emits_sse_events(tmp_path):
    client = _client(tmp_path)
    with client.stream(
        "POST", "/generate/stream", json={"prompt": "hi", "max_new_tokens": 5}
    ) as response:
        assert response.status_code == 200
        content = "".join(response.iter_text())
    assert "data:" in content
    assert "[DONE]" in content
