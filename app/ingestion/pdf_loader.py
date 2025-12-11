"""PDF and document loading utilities for medical documents."""

import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import pdfplumber
from PyPDF2 import PdfReader


@dataclass
class DocumentMetadata:
    """Metadata extracted from a medical document."""
    source_file: str
    page_count: int
    doc_type: str = "discharge_summary"
    patient_id: Optional[str] = None
    date: Optional[str] = None
    sections_found: list[str] = field(default_factory=list)


@dataclass
class LoadedDocument:
    """Container for loaded document content and metadata."""
    content: str
    metadata: DocumentMetadata
    raw_pages: list[str] = field(default_factory=list)


class MedicalPDFLoader:
    """Loader for medical PDFs with preprocessing for EHR documents."""
    
    SECTION_PATTERNS = [
        r"(?i)(discharge\s+instructions?)",
        r"(?i)(medications?\s+(?:list|instructions?|prescribed)?)",
        r"(?i)(follow[- ]?up\s+(?:care|appointments?|instructions?)?)",
        r"(?i)(diagnosis|diagnoses)",
        r"(?i)(procedures?\s+performed)",
        r"(?i)(diet(?:ary)?\s+(?:instructions?|restrictions?)?)",
        r"(?i)(activity\s+(?:restrictions?|instructions?|level)?)",
        r"(?i)(warning\s+signs?|when\s+to\s+(?:call|seek))",
        r"(?i)(allergies)",
        r"(?i)(vital\s+signs?)",
    ]
    
    def __init__(self, use_pdfplumber: bool = True):
        self.use_pdfplumber = use_pdfplumber
    
    def load(self, file_path: str | Path) -> LoadedDocument:
        """Load a PDF file and extract text with metadata."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        if self.use_pdfplumber:
            raw_pages, page_count = self._load_with_pdfplumber(file_path)
        else:
            raw_pages, page_count = self._load_with_pypdf2(file_path)
        
        combined_text = "\n\n".join(raw_pages)
        cleaned_text = self._preprocess_medical_text(combined_text)
        
        sections = self._identify_sections(cleaned_text)
        doc_type = self._classify_document(cleaned_text)
        
        metadata = DocumentMetadata(
            source_file=str(file_path.name),
            page_count=page_count,
            doc_type=doc_type,
            sections_found=sections
        )
        
        return LoadedDocument(
            content=cleaned_text,
            metadata=metadata,
            raw_pages=raw_pages
        )
    
    def _load_with_pdfplumber(self, file_path: Path) -> tuple[list[str], int]:
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = self._table_to_text(table)
                        text += f"\n{table_text}"
                pages.append(text)
            page_count = len(pdf.pages)
        return pages, page_count
    
    def _load_with_pypdf2(self, file_path: Path) -> tuple[list[str], int]:
        pages = []
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return pages, len(reader.pages)
    
    def _table_to_text(self, table: list[list]) -> str:
        if not table:
            return ""
        rows = []
        for row in table:
            cells = [str(cell).strip() for cell in row if cell]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    
    def _preprocess_medical_text(self, text: str) -> str:
        text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?m)^.*CONFIDENTIAL.*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" +\n", "\n", text)
        
        abbreviations = {
            r"\bprn\b": "PRN (as needed)",
            r"\bqd\b": "once daily",
            r"\bbid\b": "twice daily",
            r"\btid\b": "three times daily",
            r"\bqid\b": "four times daily",
            r"\bpo\b": "by mouth",
            r"\bstat\b": "immediately",
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _identify_sections(self, text: str) -> list[str]:
        found = []
        for pattern in self.SECTION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                found.append(match.group(1).strip().lower())
        return list(set(found))
    
    def _classify_document(self, text: str) -> str:
        text_lower = text.lower()
        if "discharge" in text_lower and ("instructions" in text_lower or "summary" in text_lower):
            return "discharge_summary"
        elif "medication" in text_lower and ("guide" in text_lower or "information" in text_lower):
            return "medication_guide"
        elif "progress note" in text_lower or "clinical note" in text_lower:
            return "doctor_note"
        elif "lab" in text_lower and "result" in text_lower:
            return "lab_results"
        else:
            return "medical_document"


def load_documents_from_directory(
    directory: str | Path,
    extensions: list[str] = [".pdf"]
) -> list[LoadedDocument]:
    directory = Path(directory)
    loader = MedicalPDFLoader()
    documents = []
    
    for ext in extensions:
        for file_path in directory.glob(f"*{ext}"):
            try:
                doc = loader.load(file_path)
                documents.append(doc)
                print(f"✓ Loaded: {file_path.name} ({doc.metadata.page_count} pages)")
            except Exception as e:
                print(f"✗ Failed to load {file_path.name}: {e}")
    
    return documents