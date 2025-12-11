# new fallback pipeline with llamaparser used as fallback parser
import os
import re
import logging
import pdfplumber
import pytesseract
from hashlib import md5
from typing import List, Dict
from llama_parse import LlamaParse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import PdfExtractionResult, PageText
from config import settings

logger = logging.getLogger(__name__)

LLAMAPARSE_API_KEY = settings.LLAMAPARSE_API_KEY
# Optional: Set Tesseract path if on Windows
if os.name == 'nt':
    default_tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(default_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract_path

# ===========================
# === Parsing & Extraction ===
# ===========================

def extract(filename: str) -> PdfExtractionResult:
    if os.path.splitext(filename)[1].lower() != ".pdf":
        raise ValueError("Unsupported file type")

    pages_data = []
    fallback_triggered = False

    try:
        with pdfplumber.open(filename) as pdf:
            logger.info(f"Total pages in PDF: {len(pdf.pages)}")

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = ""
                method_used = ""

                # Step 1: Try direct text extraction
                try:
                    text = page.extract_text(layout=True)
                    if text and len(text.strip()) >= 50:
                        logger.info(f"✅ Parsed text from page {page_num} using pdfplumber.")
                        method_used = "pdfplumber"
                    else:
                        raise ValueError("Text too short or missing")
                except Exception as e:
                    logger.warning(f"⚠️ pdfplumber extract_text failed for page {page_num}: {e}")
                    text = ""

                # Step 2: Try OCR using page.to_image()
                if not text:
                    try:
                        logger.info(f"🔁 Attempting OCR with pdfplumber.to_image on page {page_num}...")
                        image = page.to_image(resolution=300).original
                        text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
                        if text and len(text.strip()) >= 50:
                            logger.info(f"✅ OCR succeeded for page {page_num} with pdfplumber.to_image.")
                            method_used = "ocr:plumber"
                        else:
                            raise ValueError("OCR result too short")
                    except Exception as e:
                        logger.warning(f"⚠️ OCR via pdfplumber.to_image failed for page {page_num}: {e}")
                        text = ""

                # Extract tables (only from pdfplumber)
                table_text = ""
                try:
                    tables = page.extract_tables() or []
                    for table in tables:
                        table_text += "\n\nTable:\n" + "\n".join(
                            " | ".join(str(cell) if cell is not None else "" for cell in row)
                            for row in table
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Table extraction failed on page {page_num}: {e}")

                # Clean and combine
                cleaned_text = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
                full_text = cleaned_text + table_text

                if full_text.strip():
                    pages_data.append(PageText(
                        page_number=page_num,
                        text=full_text,
                        filename=os.path.basename(filename)
                    ))

    except Exception as e:
        logger.error(f"❌ Error using pdfplumber combo pipeline: {e}")
        fallback_triggered = True

    # If no valid pages parsed, fallback to LLaMAParse
    if not pages_data or fallback_triggered:
        logger.info("⚠️ Falling back to LLaMAParse due to insufficient content or failure.")
        try:
            parser = LlamaParse(api_key=LLAMAPARSE_API_KEY)
            job = parser.parse(filename)
            documents = job.get_text_documents()

            for i, doc in enumerate(documents):
                pages_data.append(PageText(
                    page_number=i + 1,
                    text=doc.text,
                    filename=os.path.basename(filename)
                ))

            logger.info("✅ Fallback to LLaMAParse succeeded.")
        except Exception as e:
            logger.error(f"❌ LLaMAParse fallback failed: {e}")

    return PdfExtractionResult(pages=pages_data)

# ===========================
# === Chunking Logic =========
# ===========================

def create_table_chunk(table: Dict, source: str) -> List[Dict]:
    header = table["header"]
    rows = table["rows"]
    header = [str(cell) if cell is not None else '' for cell in header]
    rows = [[str(cell) if cell is not None else '' for cell in row] for row in rows]

    # Skip incomplete tables
    if len(header) <= 1 or not any(row for row in rows if any(cell.strip() for cell in row)):
        logger.warning(f"Skipping incomplete table on pages {table['pages']}: {header}")
        return []

    md_table = "| " + " | ".join(header) + " |\n"
    md_table += "|" + "-|" * len(header) + "\n"
    for row in rows:
        md_table += "| " + " | ".join(row) + " |\n"
    content = md_table.strip()

    # Ensure valid page numbers
    valid_pages = [p for p in table["pages"] if isinstance(p, int) and p > 0]
    if not valid_pages:
        valid_pages = [1]
    page_num = min(valid_pages)

    # Split large tables
    MAX_CHUNK_SIZE = 10000
    if len(content) > MAX_CHUNK_SIZE:
        split_chunks = []
        mid_point = len(rows) // 2
        for part_rows in [rows[:mid_point], rows[mid_point:]]:
            part_table = "| " + " | ".join(header) + " |\n"
            part_table += "|" + "-|" * len(header) + "\n"
            for row in part_rows:
                part_table += "| " + " | ".join(row) + " |\n"
            split_chunks.append(part_table.strip())
        return [{
            "page_content": chunk,
            "metadata": {
                "page_number": page_num,
                "source": source or "unknown",
                "type": "table"
            }
        } for chunk in split_chunks]
    else:
        return [{
            "page_content": content,
            "metadata": {
                "page_number": page_num,
                "source": source or "unknown",
                "type": "table"
            }
        }]

def normalize_content(content: str) -> str:
    """Normalize content for deduplication by removing extra whitespace and converting to lowercase."""
    return re.sub(r'\s+', ' ', content.strip().lower())

def chunk_pdfplumber_parsed_data(pages: List[Dict]) -> List[Dict]:
    chunks = []
    seen_content = set()
    current_table = None
    text_limit = 1500
    flex_limit = 1600

    for page in pages:
        page_num = page["metadata"]["page_number"]
        source = page["metadata"]["source"]

        # Process tables
        for table in page.get("tables", []):
            if not isinstance(table, list) or not table or not isinstance(table[0], list):
                continue
            if table[0]:
                header = tuple(str(cell) if cell is not None else '' for cell in table[0])
                if current_table and current_table["header"] == header and current_table["pages"][-1] + 1 == page_num:
                    current_table["rows"].extend(table[1:])
                    current_table["pages"].append(page_num)
                else:
                    if current_table:
                        table_chunks = create_table_chunk(current_table, source)
                        for chunk in table_chunks:
                            content_hash = md5(normalize_content(chunk["page_content"]).encode()).hexdigest()
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                chunks.append(chunk)
                                logger.info(f"Created table chunk: {chunk['page_content'][:50]}... with metadata: {chunk['metadata']}")
                            else:
                                logger.warning(f"Skipped duplicate table chunk: {chunk['page_content'][:50]}...")
                    current_table = {
                        "header": header,
                        "rows": table[1:],
                        "pages": [page_num]
                    }

        # Process text
        text = page["text"]
        if text:
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para:
                    lower_para = para.lower()
                    section_type = (
                        "illustration" if "illustration" in lower_para else
                        "exhibit" if "exhibit" in lower_para else
                        "question" if any(q in lower_para for q in ["question", "questions for practice"]) else
                        "text"
                    )

                    if len(para) <= flex_limit:
                        content_hash = md5(normalize_content(para).encode()).hexdigest()
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            chunks.append({
                                "page_content": para,
                                "metadata": {
                                    "page_number": page_num,
                                    "source": source or "unknown",
                                    "type": section_type
                                }
                            })
                            logger.info(f"Created text chunk: {para[:50]}... with metadata: page={page_num}, type={section_type}")
                        else:
                            logger.warning(f"Skipped duplicate text chunk: {para[:50]}...")
                    else:
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        temp_chunk = ""
                        for sentence in sentences:
                            if len(temp_chunk) + len(sentence) <= text_limit:
                                temp_chunk += sentence + " "
                            else:
                                if temp_chunk:
                                    content_hash = md5(normalize_content(temp_chunk.strip()).encode()).hexdigest()
                                    if content_hash not in seen_content:
                                        seen_content.add(content_hash)
                                        chunks.append({
                                            "page_content": temp_chunk.strip(),
                                            "metadata": {
                                                "page_number": page_num,
                                                "source": source or "unknown",
                                                "type": section_type
                                            }
                                        })
                                        logger.info(f"Created text chunk: {temp_chunk[:50]}... with metadata: page={page_num}, type={section_type}")
                                    else:
                                        logger.warning(f"Skipped duplicate text chunk: {temp_chunk[:50]}...")
                                    temp_chunk = sentence + " "
                        if temp_chunk:
                            content_hash = md5(normalize_content(temp_chunk.strip()).encode()).hexdigest()
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                chunks.append({
                                    "page_content": temp_chunk.strip(),
                                    "metadata": {
                                        "page_number": page_num,
                                        "source": source or "unknown",
                                        "type": section_type
                                    }
                                })
                                logger.info(f"Created text chunk: {temp_chunk[:50]}... with metadata: page={page_num}, type={section_type}")
                            else:
                                logger.warning(f"Skipped duplicate text chunk: {temp_chunk[:50]}...")

    # Finalize any remaining table
    if current_table:
        table_chunks = create_table_chunk(current_table, pages[0]["metadata"]["source"])
        for chunk in table_chunks:
            content_hash = md5(normalize_content(chunk["page_content"]).encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                chunks.append(chunk)
                logger.info(f"Created table chunk: {chunk['page_content'][:50]}... with metadata: {chunk['metadata']}")
            else:
                logger.warning(f"Skipped duplicate table chunk: {chunk['page_content'][:50]}...")

    return chunks