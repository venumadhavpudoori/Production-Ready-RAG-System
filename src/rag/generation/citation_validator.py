"""Citation validation and provenance verification module."""

import re
from typing import List, Dict, Any, Set
from pydantic import BaseModel, Field
from src.rag.chunking import DocumentChunk


class CitationValidationResult(BaseModel):
    """Result of citation provenance verification."""

    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    citation_precision: float = 1.0
    cited_doc_ids: List[str] = Field(default_factory=list)
    cited_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    invalid_citation_tags: List[str] = Field(default_factory=list)
    has_citations: bool = False


class CitationValidator:
    """Verifies that generated citations accurately correspond to retrieved source passages."""

    # Matches [NASA-XXX-YYY §...] or [SOURCE X] or [DOC_ID]
    CITATION_REGEX = re.compile(r"\[(NASA-[A-Z0-9\-]+[^\]]*)\]")

    def validate(
        self,
        answer_text: str,
        retrieved_chunks: List[DocumentChunk],
    ) -> CitationValidationResult:
        """Scan answer text for inline citations and cross-reference with retrieved chunks."""
        matches = self.CITATION_REGEX.findall(answer_text)

        if not matches:
            # Check if answer abstained properly
            abstained = "not available" in answer_text.lower()
            return CitationValidationResult(
                total_citations=0,
                valid_citations=0,
                invalid_citations=0,
                citation_precision=1.0 if abstained else 0.0,
                has_citations=False,
            )

        valid_doc_ids: Set[str] = {c.doc_id for c in retrieved_chunks}
        valid_chunk_map: Dict[str, DocumentChunk] = {c.doc_id: c for c in retrieved_chunks}

        valid_count = 0
        invalid_count = 0
        cited_docs: Set[str] = set()
        cited_chunks_list: List[Dict[str, Any]] = []
        invalid_tags: List[str] = []

        for tag in matches:
            matched_valid = False
            for doc_id in valid_doc_ids:
                if doc_id in tag:
                    matched_valid = True
                    cited_docs.add(doc_id)
                    chunk = valid_chunk_map.get(doc_id)
                    if chunk and chunk.chunk_id not in [x.get("chunk_id") for x in cited_chunks_list]:
                        cited_chunks_list.append(
                            {
                                "chunk_id": chunk.chunk_id,
                                "doc_id": chunk.doc_id,
                                "title": chunk.title,
                                "section": chunk.section_heading,
                                "source_url": chunk.metadata.get("source_url", ""),
                            }
                        )
                    break

            if matched_valid:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_tags.append(tag)

        total = len(matches)
        precision = (valid_count / total) if total > 0 else 0.0

        return CitationValidationResult(
            total_citations=total,
            valid_citations=valid_count,
            invalid_citations=invalid_count,
            citation_precision=round(precision, 4),
            cited_doc_ids=sorted(list(cited_docs)),
            cited_chunks=cited_chunks_list,
            invalid_citation_tags=invalid_tags,
            has_citations=True,
        )

