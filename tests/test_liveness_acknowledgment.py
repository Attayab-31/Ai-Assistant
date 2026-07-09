"""Liveness acknowledgement detection after silence nudges."""

import pytest

from app.core.conversation import is_liveness_acknowledgment


@pytest.mark.parametrize(
    "phrase",
    [
        "yes",
        "I'm here",
        "still here",
        "ok",
    ],
)
def test_liveness_ack_english_phrases(phrase: str):
    assert is_liveness_acknowledgment(phrase, language_code="en")


@pytest.mark.parametrize(
    "phrase",
    [
        "si",
        "sí",
        "aquí estoy",
        "aqui estoy",
        "presente",
        "listo",
        "vale",
        "estoy aqui",
    ],
)
def test_liveness_ack_spanish_phrases(phrase: str):
    assert is_liveness_acknowledgment(phrase, language_code="es")


def test_liveness_ack_spanish_allows_bilingual_yes():
    assert is_liveness_acknowledgment("yes", language_code="es")


def test_liveness_ack_not_a_screening_answer():
    assert not is_liveness_acknowledgment(
        "my name is John Smith", language_code="en"
    )
    assert not is_liveness_acknowledgment(
        "me llamo Juan Perez", language_code="es"
    )
