"""Semantic and metadata-preserving text chunker for technical reports."""

import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from src.rag.data_loader import ParsedReport


class DocumentChunk(BaseModel):
    """An individual indexed chunk carrying rich citation metadata."""

    chunk_id: str
    doc_id: str
    title: str
    section_heading: str
    citation_tag: str
    text: str
    token_estimate: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarkdownSectionChunker:
    """Splits technical documents along semantic markdown sections and paragraphs."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_report(self, report: ParsedReport) -> List[DocumentChunk]:
        """Split a parsed report into metadata-rich chunks."""
        chunks: List[DocumentChunk] = []
        body = report.body_text

        # Split on markdown headers (# or ## or ###)
        section_pattern = re.compile(r"(^#{1,3}\s+.+$)", re.MULTILINE)
        splits = section_pattern.split(body)

        current_heading = "Introduction"
        running_chunk_idx = 0

        i = 0
        while i < len(splits):
            part = splits[i].strip()
            if not part:
                i += 1
                continue

            if part.startswith("#"):
                current_heading = re.sub(r"^#+\s*", "", part).strip()
                i += 1
                continue

            section_text = part
            paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]

            current_buffer: List[str] = []
            current_len = 0

            for para in paragraphs:
                para_len = len(para)
                if current_len + para_len > self.chunk_size and current_buffer:
                    chunk_body = "\n\n".join(current_buffer)
                    chunk_obj = self._create_chunk(
                        report=report,
                        heading=current_heading,
                        chunk_idx=running_chunk_idx,
                        text=chunk_body,
                    )
                    chunks.append(chunk_obj)
                    running_chunk_idx += 1

                    # Retain last paragraph for overlap if within overlap window
                    if len(current_buffer[-1]) <= self.chunk_overlap:
                        current_buffer = [current_buffer[-1], para]
                        current_len = len(current_buffer[0]) + para_len
                    else:
                        current_buffer = [para]
                        current_len = para_len
                else:
                    current_buffer.append(para)
                    current_len += para_len

            if current_buffer:
                chunk_body = "\n\n".join(current_buffer)
                chunk_obj = self._create_chunk(
                    report=report,
                    heading=current_heading,
                    chunk_idx=running_chunk_idx,
                    text=chunk_body,
                )
                chunks.append(chunk_obj)
                running_chunk_idx += 1

            i += 1

        return chunks

    def _create_chunk(
        self, report: ParsedReport, heading: str, chunk_idx: int, text: str
    ) -> DocumentChunk:
        chunk_id = f"{report.document_id}_c{chunk_idx:03d}"
        citation_tag = f"[{report.document_id} §{heading}]"
        token_estimate = max(1, len(text) // 4)

        meta = {
            "chunk_id": chunk_id,
            "doc_id": report.document_id,
            "title": report.title,
            "author": report.author,
            "publication_year": report.publication_year,
            "mission": report.mission,
            "target_body": report.target_body,
            "landing_site": report.landing_site,
            "source_url": report.source_url,
            "section_heading": heading,
            "citation_tag": citation_tag,
            "token_estimate": token_estimate,
        }

        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=report.document_id,
            title=report.title,
            section_heading=heading,
            citation_tag=citation_tag,
            text=text,
            token_estimate=token_estimate,
            metadata=meta,
        )

