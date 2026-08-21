"""Extract plain text from uploaded resume/JD files (PDF, DOCX, TXT)."""


from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt", "md", "text"}


def extract_text(filename: str, original_name: str = None) -> str:
    """Extract text from a file based on its extension."""
    name = original_name or filename
    ext = name.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        reader = PdfReader(filename)
        return "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()

    if ext == "docx":
        doc = Document(filename)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()

    if ext in ("txt", "md", "text"):
        with open(filename, encoding="utf-8", errors="replace") as f:
            return f.read().strip()

    raise ValueError(
        f"Unsupported file type '.{ext}'. Please upload a PDF, DOCX, or TXT file."
    )
