"""Tests for sentence boundary detection."""

from wyoming_piper.sentence_boundary import SentenceBoundaryDetector


def test_one_chunk() -> None:
    sbd = SentenceBoundaryDetector()
    assert not list(sbd.add_chunk("Test chunk"))
    assert sbd.finish() == "Test chunk"


def test_one_chunk_with_punctuation() -> None:
    sbd = SentenceBoundaryDetector()
    assert list(sbd.add_chunk("Test chunk 1. Test chunk 2")) == ["Test chunk 1."]
    assert sbd.finish() == "Test chunk 2"


def test_multiple_chunks() -> None:
    sbd = SentenceBoundaryDetector()
    assert not list(sbd.add_chunk("Test chunk"))
    assert list(sbd.add_chunk(" 1. Test chunk")) == ["Test chunk 1."]
    assert not list(sbd.add_chunk(" 2."))
    assert sbd.finish() == "Test chunk 2."
