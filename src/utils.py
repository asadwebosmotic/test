# Re-export from utils package for backwards compatibility
from dataclasses import dataclass
from typing import List

@dataclass
class PageText:
    page_number: int
    text: str
    filename: str

@dataclass
class PdfExtractionResult:
    pages: List[PageText]
