"""Tests for sentence boundary detection."""

from wyoming_piper.sentence_boundary import SentenceBoundaryDetector, remove_asterisks


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


def test_numbered_lists() -> None:
    sbd = SentenceBoundaryDetector()
    sentences = list(
        sbd.add_chunk(
            "Final Fantasy VII features several key characters who drive the narrative: "
            "1. **Cloud Strife** - The protagonist, an ex-SOLDIER mercenary and a skilled fighter. "
            "2. **Aerith Gainsborough (Aeris)** - A kindhearted flower seller with spiritual powers and deep connections to the planet's ecosystem. "
            "3. **Barret Wallace** - A leader of eco-terrorists called AVALANCHE, fighting against Shinra Corporation's exploitation of the planet. "
            "4. **Tifa Lockhart** - Cloud's childhood friend who runs a bar in Sector 7 and helps him recover from past trauma. "
            "5. **Sephiroth** - The main antagonist, an ex-SOLDIER with god-like abilities, seeking to control or destroy the planet. "
            "6. **Red XIII (aka Red 13)** - A member of a catlike race called Cetra, searching for answers about his heritage and destiny. "
            "7. **Vincent Valentine** - A brooding former Turk who lives in isolation from guilt over past failures but aids Cloud's party with his powerful abilities. "
            "8. **Cid Highwind** - The pilot of the rocket plane Highwind and a skilled engineer working on various airship projects. 9. "
            "**Shinra Employees (JENOVA Project)** - Characters like Professor Hojo, President Shinra, and Reno who play crucial roles in the plot's development. "
            "Each character brings unique skills and perspectives to the story, contributing to its rich narrative and gameplay dynamics."
        )
    )
    assert len(sentences) == 9
    assert sbd.finish().startswith("Each character")


def test_remove_word_asterisks() -> None:
    sbd = SentenceBoundaryDetector()
    assert list(
        sbd.add_chunk(
            "**Test** sentence with *emphasized* words! Another *** sentence."
        )
    ) == ["Test sentence with emphasized words!"]
    assert sbd.finish() == "Another *** sentence."


def test_remove_line_asterisks() -> None:
    assert (
        remove_asterisks("* Test item 1.\n\n** Test item 2\n * Test item 3.")
        == " Test item 1.\n\n Test item 2\n Test item 3."
    )
