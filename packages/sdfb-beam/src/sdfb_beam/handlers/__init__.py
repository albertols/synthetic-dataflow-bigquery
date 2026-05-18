"""`ModelClient` / `ModelHandler` implementations.

Production:
  - `vllm_handler.py` / `vllm_client.py` — vLLM-backed (M1 §9, M4-only).

Test / development:
  - `fake_client.py` — deterministic `ModelClient` for laptop / DirectRunner.
"""

from sdfb_beam.handlers.fake_client import FakeModelClient

__all__ = ["FakeModelClient"]
