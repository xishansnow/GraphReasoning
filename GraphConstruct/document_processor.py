"""
Document Processing Module
==========================

Handles loading and preprocessing of historical documents from various formats.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    """Represents a processed document."""
    
    content: str
    source: str
    document_type: str  # 'text', 'pdf', 'markdown'
    metadata: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "document_type": self.document_type,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class DocumentProcessor:
    """Processes various document formats for knowledge graph construction."""
    
    def __init__(self, max_size_mb: int = 100):
        """
        Initialize document processor.
        
        Args:
            max_size_mb: Maximum file size in MB (default: 100)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.supported_formats = {'.txt', '.pdf', '.md', '.markdown'}
    
    def load_documents(self, source: Union[str, Path], 
                      file_types: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from directory or file.
        
        Args:
            source: Directory path or file path
            file_types: List of file extensions to load (e.g., ['.txt', '.pdf'])
        
        Returns:
            List of Document objects
        """
        source_path = Path(source)
        documents = []
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source}")
        
        # Determine which formats to load
        formats = set(file_types) if file_types else self.supported_formats
        
        if source_path.is_file():
            # Load single file
            if source_path.suffix in formats:
                doc = self._load_file(source_path)
                if doc:
                    documents.append(doc)
        else:
            # Load all matching files from directory
            for file_path in source_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in formats:
                    doc = self._load_file(file_path)
                    if doc:
                        documents.append(doc)
        
        return documents
    
    def _load_file(self, file_path: Path) -> Optional[Document]:
        """
        Load a single file.
        
        Args:
            file_path: Path to file
        
        Returns:
            Document object or None if loading fails
        """
        # Check file size
        if file_path.stat().st_size > self.max_size_bytes:
            print(f"Warning: File too large, skipping: {file_path}")
            return None
        
        try:
            if file_path.suffix == '.pdf':
                return self._load_pdf(file_path)
            elif file_path.suffix in {'.txt', '.md', '.markdown'}:
                return self._load_text(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        
        return None
    
    def _load_text(self, file_path: Path) -> Document:
        """Load text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Document(
            content=content,
            source=str(file_path),
            document_type='text' if file_path.suffix == '.txt' else 'markdown',
            metadata={
                'filename': file_path.name,
                'file_size': file_path.stat().st_size,
                'encoding': 'utf-8'
            },
            created_at=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
    
    def _load_pdf(self, file_path: Path) -> Optional[Document]:
        """Load PDF file."""
        try:
            import PyPDF2
        except ImportError:
            print("PyPDF2 not installed. Install with: pip install PyPDF2")
            return None
        
        try:
            content_parts = []
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        content_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            
            content = "\n".join(content_parts)
            
            return Document(
                content=content,
                source=str(file_path),
                document_type='pdf',
                metadata={
                    'filename': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'num_pages': num_pages
                },
                created_at=datetime.fromtimestamp(file_path.stat().st_mtime)
            )
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return None
    
    def split_into_chunks(self, document: Document, 
                         chunk_size: int = 500, 
                         overlap: int = 50) -> List[str]:
        """
        Split document into overlapping chunks.
        
        Args:
            document: Document to split
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks
        
        Returns:
            List of text chunks
        """
        content = document.content
        chunks = []
        
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk = content[start:end]
            chunks.append(chunk.strip())
            
            # Move start position, accounting for overlap
            start = end - overlap
        
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text (remove extra whitespace, normalize).
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special control characters
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\t')
        
        return text.strip()
