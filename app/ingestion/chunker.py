"""Text chunking utilities optimized for medical documents."""

import re
from dataclasses import dataclass, field
from typing import Optional
import tiktoken


@dataclass
class TextChunk:
    """A chunk of text with metadata for retrieval."""
    content: str
    chunk_id: str
    source_file: str
    chunk_index: int
    token_count: int
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "section": self.section or "unknown",
            **self.metadata
        }


class MedicalTextChunker:
    """Chunker optimized for medical documents."""
    
    SECTION_HEADERS = [
        r"(?i)^#{1,3}\s+",
        r"(?i)^(?:DISCHARGE INSTRUCTIONS|MEDICATIONS?|FOLLOW[- ]?UP|DIAGNOSIS|"
        r"PROCEDURES?|DIET|ACTIVITY|WARNING SIGNS?|ALLERGIES|VITAL SIGNS?|"
        r"INSTRUCTIONS?|CARE PLAN|TREATMENT|PRECAUTIONS?)\s*:?\s*$",
    ]
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, model_name: str = "gpt-4o-mini"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def chunk_document(self, document) -> list[TextChunk]:
        sections = self._split_into_sections(document.content)
        all_chunks = []
        chunk_index = 0
        
        for section_name, section_text in sections:
            section_chunks = self._chunk_section(
                section_text, section_name, document.metadata.source_file, start_index=chunk_index
            )
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return all_chunks
    
    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        lines = text.split("\n")
        sections = []
        current_section = "introduction"
        current_content = []
        
        for line in lines:
            is_header = any(re.match(p, line.strip()) for p in self.SECTION_HEADERS)
            if is_header and current_content:
                sections.append((current_section, "\n".join(current_content).strip()))
                current_section = self._normalize_section_name(line)
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections.append((current_section, "\n".join(current_content).strip()))
        
        return [(n, c) for n, c in sections if c.strip()]
    
    def _normalize_section_name(self, header: str) -> str:
        name = re.sub(r"^#+\s*", "", header)
        name = re.sub(r"[:\-_]+$", "", name)
        return name.strip().lower()
    
    def _chunk_section(self, text: str, section_name: str, source_file: str, start_index: int = 0) -> list[TextChunk]:
        if not text.strip():
            return []
        
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(" ".join(current_chunk), section_name, source_file, start_index + len(chunks)))
                    current_chunk, current_tokens = [], 0
                for sub in self._split_long_sentence(sentence):
                    chunks.append(self._create_chunk(sub, section_name, source_file, start_index + len(chunks)))
                continue
            
            if current_tokens + sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(" ".join(current_chunk), section_name, source_file, start_index + len(chunks)))
                overlap = self._get_overlap_sentences(current_chunk, self.chunk_overlap)
                current_chunk = overlap + [sentence]
                current_tokens = self.count_tokens(" ".join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(self._create_chunk(" ".join(current_chunk), section_name, source_file, start_index + len(chunks)))
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        protected = text
        for abbr in ["Dr.", "Mr.", "Mrs.", "Ms.", "mg.", "mL.", "oz.", "lb.", "kg.", "a.m.", "p.m.", "e.g.", "i.e.", "vs."]:
            protected = protected.replace(abbr, abbr.replace(".", "<<DOT>>"))
        sentences = re.split(r"(?<=[.!?])\s+", protected)
        return [s.replace("<<DOT>>", ".").strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str) -> list[str]:
        parts = re.split(r"[,;]|\s+(?:and|or|but)\s+", sentence)
        chunks, current, current_tokens = [], [], 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            pt = self.count_tokens(part)
            if current_tokens + pt > self.chunk_size:
                if current:
                    chunks.append(", ".join(current))
                current, current_tokens = [part], pt
            else:
                current.append(part)
                current_tokens += pt
        if current:
            chunks.append(", ".join(current))
        return chunks
    
    def _get_overlap_sentences(self, sentences: list[str], target_tokens: int) -> list[str]:
        overlap, tokens = [], 0
        for s in reversed(sentences):
            st = self.count_tokens(s)
            if tokens + st > target_tokens:
                break
            overlap.insert(0, s)
            tokens += st
        return overlap
    
    def _create_chunk(self, content: str, section: str, source_file: str, index: int) -> TextChunk:
        chunk_id = re.sub(r"[^\w\-_]", "_", f"{source_file}_{section}_{index}")
        return TextChunk(content=content, chunk_id=chunk_id, source_file=source_file,
                         chunk_index=index, token_count=self.count_tokens(content), section=section)


def chunk_documents(documents: list, chunk_size: int = 512, chunk_overlap: int = 50) -> list[TextChunk]:
    chunker = MedicalTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"✓ Chunked {doc.metadata.source_file}: {len(chunks)} chunks")
    return all_chunks