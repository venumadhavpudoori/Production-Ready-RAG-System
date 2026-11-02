"""Document loader and parser for NASA technical reports."""

import hashlib
from pathlib import Path
from typing import Dict, List, Any
import re
from pydantic import BaseModel, Field


class ParsedReport(BaseModel):
    """Structured representation of an ingested NASA technical report."""

    document_id: str
    title: str
    author: str = "NASA"
    publication_year: int = 2021
    mission: str = "Mars Exploration"
    target_body: str = "Mars"
    landing_site: str = "Unspecified"
    source_url: str = ""
    file_path: str
    content_hash: str
    body_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NASAReportLoader:
    """Loads and extracts metadata from NASA technical reports."""

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)

    def _parse_frontmatter(self, text: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML-like frontmatter metadata from markdown document."""
        metadata: Dict[str, Any] = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter_raw = parts[1].strip()
                body = parts[2].strip()
                for line in frontmatter_raw.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        clean_key = k.strip()
                        clean_val = v.strip().strip("'\"")
                        if clean_key == "publication_year":
                            try:
                                metadata[clean_key] = int(clean_val)
                            except ValueError:
                                metadata[clean_key] = 2021
                        else:
                            metadata[clean_key] = clean_val
        return metadata, body

    def load_document(self, file_path: Path) -> ParsedReport:
        """Parse a single NASA markdown technical report."""
        text = file_path.read_text(encoding="utf-8")
        metadata, body = self._parse_frontmatter(text)

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        doc_id = metadata.get("document_id", file_path.stem)
        title = metadata.get("title", file_path.stem.replace("_", " ").title())

        full_meta = {
            "document_id": doc_id,
            "title": title,
            "author": metadata.get("author", "NASA"),
            "publication_year": metadata.get("publication_year", 2021),
            "mission": metadata.get("mission", "Mars Exploration"),
            "target_body": metadata.get("target_body", "Mars"),
            "landing_site": metadata.get("landing_site", "Unspecified"),
            "source_url": metadata.get("source_url", ""),
            "file_path": str(file_path.resolve()),
            "content_hash": content_hash,
        }

        return ParsedReport(
            document_id=doc_id,
            title=title,
            author=full_meta["author"],
            publication_year=full_meta["publication_year"],
            mission=full_meta["mission"],
            target_body=full_meta["target_body"],
            landing_site=full_meta["landing_site"],
            source_url=full_meta["source_url"],
            file_path=str(file_path.resolve()),
            content_hash=content_hash,
            body_text=body,
            metadata=full_meta,
        )

    def load_all_reports(self) -> List[ParsedReport]:
        """Scan raw directory and parse all available technical reports."""
        if not self.raw_data_dir.exists():
            return []
        md_files = sorted(self.raw_data_dir.glob("*.md"))
        return [self.load_document(f) for f in md_files]

