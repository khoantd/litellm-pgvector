from typing import List, Optional
from fastapi import UploadFile
import io
import logging
import anyio

logger = logging.getLogger(__name__)


class ExtractedPart:
    """Represents a part of extracted text with metadata"""
    def __init__(self, text: str, metadata: Optional[dict] = None):
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self):
        return f"ExtractedPart(text_length={len(self.text)}, metadata={self.metadata})"


async def extract_text_from_upload(file: UploadFile, delimiter: str = "\n", sheet: Optional[str] = None) -> List[ExtractedPart]:
    """
    Extract text from an uploaded file based on its MIME type or extension.
    
    Args:
        file: FastAPI UploadFile object
        delimiter: Delimiter for CSV/XLSX row concatenation (default: "\n")
        sheet: Sheet name for XLSX files (default: first sheet)
        
    Returns:
        List of ExtractedPart objects with text and metadata
    """
    content = await file.read()
    file_name = file.filename or "unknown"
    content_type = file.content_type or ""
    
    # Determine file type from extension or MIME type
    file_ext = file_name.lower().split('.')[-1] if '.' in file_name else ""
    
    # Try PDF
    if file_ext == "pdf" or content_type == "application/pdf":
        return await extract_from_pdf(content, file_name)
    
    # Try DOCX
    if file_ext == "docx" or content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return await extract_from_docx(content, file_name)
    
    # Try TXT
    if file_ext == "txt" or content_type in ["text/plain", "text/markdown"]:
        return await extract_from_txt(content, file_name)
    
    # Try CSV
    if file_ext == "csv" or content_type == "text/csv":
        return await extract_from_csv(content, file_name, delimiter)
    
    # Try XLSX
    if file_ext == "xlsx" or content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return await extract_from_xlsx(content, file_name, sheet, delimiter)
    
    raise ValueError(f"Unsupported file type: {file_name} (extension: {file_ext}, content_type: {content_type})")


async def extract_from_pdf(content: bytes, file_name: str) -> List[ExtractedPart]:
    """Extract text from PDF file"""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required for PDF extraction. Install with: pip install pypdf")
    
    def _extract_pdf_sync():
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        
        parts = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                parts.append(ExtractedPart(
                    text=text,
                    metadata={
                        "source": "pdf",
                        "file_name": file_name,
                        "page": page_num,
                        "mime_type": "application/pdf"
                    }
                ))
        
        if not parts:
            raise ValueError("No text could be extracted from PDF")
        
        return parts
    
    try:
        parts = await anyio.to_thread.run_sync(_extract_pdf_sync)
        return parts
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


async def extract_from_docx(content: bytes, file_name: str) -> List[ExtractedPart]:
    """Extract text from DOCX file"""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX extraction. Install with: pip install python-docx")
    
    def _extract_docx_sync():
        docx_file = io.BytesIO(content)
        doc = Document(docx_file)
        
        # Extract all paragraphs
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        
        if not paragraphs:
            raise ValueError("No text could be extracted from DOCX")
        
        # Combine all paragraphs into a single part
        full_text = "\n\n".join(paragraphs)
        return [ExtractedPart(
            text=full_text,
            metadata={
                "source": "docx",
                "file_name": file_name,
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "paragraph_count": len(paragraphs)
            }
        )]
    
    try:
        parts = await anyio.to_thread.run_sync(_extract_docx_sync)
        return parts
    except Exception as e:
        logger.error(f"Failed to extract DOCX: {e}")
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")


async def extract_from_txt(content: bytes, file_name: str) -> List[ExtractedPart]:
    """Extract text from TXT file with encoding detection"""
    try:
        import chardet
    except ImportError:
        raise ImportError("chardet is required for TXT extraction. Install with: pip install chardet")
    
    def _extract_txt_sync():
        # Detect encoding
        detected = chardet.detect(content)
        encoding = detected.get('encoding', 'utf-8') if detected else 'utf-8'
        
        # Try to decode with detected encoding, fallback to utf-8
        try:
            text = content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            text = content.decode('utf-8', errors='replace')
        
        if not text.strip():
            raise ValueError("File appears to be empty")
        
        return [ExtractedPart(
            text=text,
            metadata={
                "source": "txt",
                "file_name": file_name,
                "mime_type": "text/plain",
                "encoding": encoding
            }
        )]
    
    try:
        parts = await anyio.to_thread.run_sync(_extract_txt_sync)
        return parts
    except Exception as e:
        logger.error(f"Failed to extract TXT: {e}")
        raise ValueError(f"Failed to extract text from TXT: {str(e)}")


async def extract_from_csv(content: bytes, file_name: str, delimiter: str = "\n") -> List[ExtractedPart]:
    """Extract text from CSV file, treating entire file as a single document"""
    try:
        import pandas as pd
        import chardet
    except ImportError:
        raise ImportError("pandas and chardet are required for CSV extraction. Install with: pip install pandas chardet")
    
    def _extract_csv_sync():
        # Detect encoding
        detected = chardet.detect(content)
        encoding = detected.get('encoding', 'utf-8') if detected else 'utf-8'
        
        # Read CSV
        csv_file = io.BytesIO(content)
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            csv_file.seek(0)
            df = pd.read_csv(csv_file, encoding='utf-8', errors='replace')
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Convert all rows to strings and concatenate with delimiter
        text_parts = []
        for idx, row in df.iterrows():
            row_str = delimiter.join(str(val) for val in row.values if pd.notna(val))
            if row_str.strip():
                text_parts.append(row_str)
        
        full_text = delimiter.join(text_parts)
        
        if not full_text.strip():
            raise ValueError("No valid content could be extracted from CSV")
        
        return [ExtractedPart(
            text=full_text,
            metadata={
                "source": "csv",
                "file_name": file_name,
                "mime_type": "text/csv",
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "delimiter": delimiter
            }
        )]
    
    try:
        parts = await anyio.to_thread.run_sync(_extract_csv_sync)
        return parts
    except Exception as e:
        logger.error(f"Failed to extract CSV: {e}")
        raise ValueError(f"Failed to extract text from CSV: {str(e)}")


async def extract_from_xlsx(content: bytes, file_name: str, sheet: Optional[str] = None, delimiter: str = "\n") -> List[ExtractedPart]:
    """Extract text from XLSX file, treating entire sheet as a single document"""
    try:
        import pandas as pd
        import openpyxl
    except ImportError:
        raise ImportError("pandas and openpyxl are required for XLSX extraction. Install with: pip install pandas openpyxl")
    
    def _extract_xlsx_sync():
        xlsx_file = io.BytesIO(content)
        excel_file = pd.ExcelFile(xlsx_file, engine='openpyxl')
        
        # Select sheet
        sheet_name = sheet if sheet and sheet in excel_file.sheet_names else excel_file.sheet_names[0]
        
        # Read selected sheet
        df = pd.read_excel(xlsx_file, sheet_name=sheet_name, engine='openpyxl')
        
        if df.empty:
            raise ValueError(f"Sheet '{sheet_name}' is empty")
        
        # Convert all rows to strings and concatenate with delimiter
        text_parts = []
        for idx, row in df.iterrows():
            row_str = delimiter.join(str(val) for val in row.values if pd.notna(val))
            if row_str.strip():
                text_parts.append(row_str)
        
        full_text = delimiter.join(text_parts)
        
        if not full_text.strip():
            raise ValueError(f"No valid content could be extracted from sheet '{sheet_name}'")
        
        return [ExtractedPart(
            text=full_text,
            metadata={
                "source": "xlsx",
                "file_name": file_name,
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "sheet": sheet_name,
                "available_sheets": excel_file.sheet_names,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "delimiter": delimiter
            }
        )]
    
    try:
        parts = await anyio.to_thread.run_sync(_extract_xlsx_sync)
        return parts
    except Exception as e:
        logger.error(f"Failed to extract XLSX: {e}")
        raise ValueError(f"Failed to extract text from XLSX: {str(e)}")

