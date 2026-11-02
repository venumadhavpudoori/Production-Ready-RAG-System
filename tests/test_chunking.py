"""Unit tests for document parsing and chunking."""

from pathlib import Path
from src.rag.data_loader import NASAReportLoader, ParsedReport
from src.rag.chunking import MarkdownSectionChunker, DocumentChunk


def test_data_loader_frontmatter_parsing(tmp_path: Path):
    doc_content = """---
document_id: TEST-DOC-001
title: Test Mission Report
author: NASA Test Team
publication_year: 2024
mission: Mars Test Mission
---

# Test Heading
This is a test paragraph describing the flight telemetry of the vehicle.
"""
    doc_path = tmp_path / "test_doc.md"
    doc_path.write_text(doc_content, encoding="utf-8")

    loader = NASAReportLoader(tmp_path)
    report = loader.load_document(doc_path)

    assert report.document_id == "TEST-DOC-001"
    assert report.title == "Test Mission Report"
    assert report.publication_year == 2024
    assert report.mission == "Mars Test Mission"
    assert "Test Heading" in report.body_text


def test_chunker_metadata_preservation():
    report = ParsedReport(
        document_id="TEST-DOC-002",
        title="Sample Rover Telemetry",
        author="JPL",
        publication_year=2021,
        mission="Mars 2020",
        target_body="Mars",
        landing_site="Jezero",
        source_url="https://nasa.gov/test",
        file_path="/tmp/test.md",
        content_hash="abc12345",
        body_text="""## 1. Landing Operations
The lander touched down safely on the plains.

## 2. Thermal Telemetry
The peak heatshield temperature reached 1500 C during hypersonic entry.
""",
    )

    chunker = MarkdownSectionChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_report(report)

    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.doc_id == "TEST-DOC-002"
        assert c.title == "Sample Rover Telemetry"
        assert c.citation_tag.startswith("[TEST-DOC-002 §")
        assert c.token_estimate > 0

