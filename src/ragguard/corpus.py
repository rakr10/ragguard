"""Loading and chunking the document corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    heading: str
    text: str

    @property
    def search_text(self) -> str:
        """Text used for retrieval: the section heading carries useful context."""
        return f"{self.heading}. {self.text}" if self.heading else self.text


def load_corpus(directory: str | Path) -> list[Chunk]:
    """Split every markdown file into paragraph chunks tagged with their nearest heading."""
    directory = Path(directory)
    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md documents found in {directory}")
    chunks: list[Chunk] = []
    for path in files:
        title = heading = ""
        index = 0
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith("## "):
                heading = block[3:].strip()
            elif block.startswith("# "):
                title = block[2:].strip()
                heading = title
            else:
                index += 1
                chunks.append(Chunk(f"{path.stem}#{index}", path.stem, heading or title, block))
    return chunks
