"""
OCR Service - Ultra-Fast Hybrid Extraction
==========================================
Smart document detection with tiered processing:
1. Direct text extraction (instant) - for text-based PDFs
2. Azure Document Intelligence (fast) - for scanned documents  
3. EasyOCR fallback (slow) - when no API keys available

Performance targets:
- Text PDFs: 637 pages in <10 seconds
- Scanned PDFs: 637 pages in 8-12 minutes (with Azure)
"""

import asyncio
import os
import tempfile
import unicodedata
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import re

import fitz  # PyMuPDF
import structlog

from app.config import OCRProvider, get_settings

logger = structlog.get_logger()


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class OCRProgress:
    """Progress tracking for OCR operations."""
    def __init__(self, current_page: int, total_pages: int, phase: str = "extracting"):
        self.current_page = current_page
        self.total_pages = total_pages
        self.phase = phase


ProgressCallback = Optional[Callable[[OCRProgress], None]]


# =============================================================================
# SMART DOCUMENT ANALYZER
# =============================================================================

class DocumentAnalyzer:
    """
    Analyzes documents to determine the best extraction method.
    Checks if PDF has extractable text or needs OCR.
    """
    
    # Minimum characters per page to consider it "text-based"
    MIN_CHARS_PER_PAGE = 100
    # Sample pages to check
    SAMPLE_PAGES = 5
    
    @staticmethod
    def analyze_pdf(file_path: str) -> dict:
        """
        Analyze a PDF to determine extraction strategy.
        
        Returns:
            {
                'has_text': bool,           # True if text can be extracted directly
                'text_coverage': float,     # Percentage of pages with text
                'total_pages': int,
                'detected_scripts': set,    # Detected scripts/languages
                'recommended_method': str,  # 'direct', 'ocr_azure', 'ocr_easyocr'
            }
        """
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            pages_with_text = 0
            all_text_sample = ""
            detected_scripts = set()
            
            # Sample pages evenly across document
            sample_indices = []
            if total_pages <= DocumentAnalyzer.SAMPLE_PAGES:
                sample_indices = list(range(total_pages))
            else:
                step = max(1, total_pages // DocumentAnalyzer.SAMPLE_PAGES)
                sample_indices = [min(i * step, total_pages - 1) for i in range(DocumentAnalyzer.SAMPLE_PAGES)]
            
            for page_num in sample_indices:
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                
                if len(text) >= DocumentAnalyzer.MIN_CHARS_PER_PAGE:
                    pages_with_text += 1
                    all_text_sample += text[:500]  # Sample for language detection
            
            doc.close()
            
            # Calculate coverage
            text_coverage = pages_with_text / len(sample_indices) if sample_indices else 0
            
            # Detect languages in sample text
            detected_scripts = DocumentAnalyzer._detect_scripts(all_text_sample)
            
            # Determine recommended method
            if text_coverage >= 0.7:  # 70%+ pages have text
                recommended = 'direct'
            else:
                # Check if Azure is available
                settings = get_settings()
                if settings.azure_doc_endpoint and settings.azure_doc_key:
                    recommended = 'ocr_azure'
                else:
                    recommended = 'ocr_easyocr'
            
            return {
                'has_text': text_coverage >= 0.7,
                'text_coverage': text_coverage,
                'total_pages': total_pages,
                'detected_scripts': detected_scripts,
                'recommended_method': recommended,
            }
            
        except Exception as e:
            logger.error("Document analysis failed", error=str(e))
            return {
                'has_text': False,
                'text_coverage': 0,
                'total_pages': 0,
                'detected_scripts': set(),
                'recommended_method': 'ocr_easyocr',
            }
    
    @staticmethod
    def _detect_scripts(text: str) -> set:
        """Detect scripts present in text sample."""
        scripts = set()
        for char in text[:1000]:  # Sample first 1000 chars
            try:
                name = unicodedata.name(char, '')
                if 'CYRILLIC' in name:
                    scripts.add('cyrillic')
                elif 'ARABIC' in name:
                    scripts.add('arabic')
                elif 'CJK' in name:
                    scripts.add('chinese')
                elif 'HIRAGANA' in name or 'KATAKANA' in name:
                    scripts.add('japanese')
                elif 'HANGUL' in name:
                    scripts.add('korean')
                elif char.isalpha():
                    scripts.add('latin')
            except:
                pass
        return scripts


# =============================================================================
# DIRECT TEXT EXTRACTION (INSTANT - For text-based PDFs)
# =============================================================================

class DirectTextExtractor:
    """
    Extracts text directly from PDFs using PyMuPDF.
    Lightning fast - 637 pages in <10 seconds.
    Works for PDFs with embedded/searchable text.

    ENHANCED: Uses layout-preserving extraction for TOC pages
    to maintain dot leaders and page number alignment.
    """

    @staticmethod
    async def extract(
        file_path: str,
        progress_callback: ProgressCallback = None
    ) -> Tuple[str, int, List[str]]:
        """
        Extract text directly from PDF.
        Returns: (text, page_count, page_texts_list)
        """
        # Run extraction in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            DirectTextExtractor._extract_sync,
            file_path,
            progress_callback
        )

    @staticmethod
    def _is_toc_page(text: str) -> bool:
        """
        Detect if a page is a Table of Contents page.
        TOC pages have: many short lines, trailing page numbers, dot leaders.
        """
        if not text or len(text) < 200:
            return False

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 8:
            return False

        # Count lines with trailing page numbers
        trailing_number_lines = sum(1 for ln in lines if re.search(r'\d{1,4}\s*$', ln))

        # Count lines with dot leaders
        dot_leader_lines = sum(1 for ln in lines if '...' in ln or '…' in ln)

        # Count short lines (typical for TOC entries)
        short_lines = sum(1 for ln in lines if len(ln) <= 80)

        # Heuristic: TOC pages have many trailing numbers and/or dot leaders
        has_many_numbers = trailing_number_lines >= max(5, len(lines) * 0.3)
        has_dot_leaders = dot_leader_lines >= 3
        mostly_short = short_lines >= len(lines) * 0.5

        return (has_many_numbers or has_dot_leaders) and mostly_short

    @staticmethod
    def _is_abbreviation_page(text: str) -> bool:
        """
        Detect if a page contains abbreviation/acronym definitions.
        Format: "ACRONYM — full expansion" or similar patterns.
        """
        if not text or len(text) < 50:
            return False
        
        text_lower = text.lower()
        
        # Check for abbreviation header keywords (Ukrainian, Russian, English)
        header_keywords = [
            'скорочення', 'скороченн',      # Ukrainian: abbreviations
            'позначення',                    # Ukrainian: designations
            'абревіатур',                    # Ukrainian: abbreviations
            'умовн',                         # Ukrainian: conventional
            'сокращени',                     # Russian: abbreviations
            'обозначени',                    # Russian: designations
            'abbreviation', 'acronym',       # English
            'legend', 'notation'             # English
        ]
        
        has_header = any(kw in text_lower for kw in header_keywords)
        
        # Count abbreviation patterns: 2-6 UPPERCASE letters followed by dash and text
        # Works with both Cyrillic (А-ЯІЇЄҐ) and Latin (A-Z)
        import re
        abbrev_pattern = re.compile(r'[A-ZА-ЯІЇЄҐ]{2,8}\s*[-—–]\s*[A-Za-zА-Яа-яІіЇїЄєҐґ]', re.UNICODE)
        abbrev_matches = len(abbrev_pattern.findall(text))
        
        # If has header and at least 3 abbreviation patterns, or 5+ patterns without header
        return (has_header and abbrev_matches >= 3) or abbrev_matches >= 5

    @staticmethod
    def _extract_with_layout(page) -> str:
        """
        Extract text preserving layout structure.
        Uses word positions to rebuild lines with proper spacing.
        Essential for TOC pages with dot leaders and page numbers.
        """
        # Get text as dictionary with word positions
        text_dict = page.get_text("dict")

        if not text_dict.get("blocks"):
            return ""

        # Collect all words with their positions
        words = []
        for block in text_dict["blocks"]:
            if block.get("type") != 0:  # Skip non-text blocks
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        words.append({
                            "text": text,
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                            "y_center": (bbox[1] + bbox[3]) / 2
                        })

        if not words:
            return ""

        # Group words by Y-position (within tolerance) to form lines
        words.sort(key=lambda w: (w["y_center"], w["x0"]))

        lines = []
        current_line = []
        current_y = None
        Y_TOLERANCE = 5  # Pixels tolerance for same line

        for word in words:
            if current_y is None:
                current_y = word["y_center"]
                current_line = [word]
            elif abs(word["y_center"] - current_y) <= Y_TOLERANCE:
                current_line.append(word)
            else:
                # Finalize current line
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                current_y = word["y_center"]

        if current_line:
            lines.append(current_line)

        # Rebuild text for each line preserving spacing
        output_lines = []
        for line_words in lines:
            # Sort words by X position (left to right)
            line_words.sort(key=lambda w: w["x0"])

            line_parts = []
            prev_x1 = None

            for word in line_words:
                if prev_x1 is not None:
                    # Calculate gap between words
                    gap = word["x0"] - prev_x1

                    # If large gap, insert spaces or dot leader
                    if gap > 50:  # Large gap - likely dot leader area
                        # Estimate number of dots based on gap
                        num_dots = max(3, int(gap / 8))
                        line_parts.append(" " + "." * num_dots + " ")
                    elif gap > 10:  # Medium gap
                        line_parts.append("  ")
                    else:  # Normal word spacing
                        line_parts.append(" ")

                line_parts.append(word["text"])
                prev_x1 = word["x1"]

            output_lines.append("".join(line_parts))

        return "\n".join(output_lines)

    @staticmethod
    def _extract_sync(
        file_path: str,
        progress_callback: ProgressCallback = None
    ) -> Tuple[str, int, List[str]]:
        """Synchronous text extraction. Returns (full_text, page_count, page_texts_list)"""
        doc = fitz.open(file_path)
        total_pages = len(doc)
        page_texts = []

        for page_num in range(total_pages):
            page = doc.load_page(page_num)

            # First, get basic text to check if it's a TOC or abbreviation page
            basic_text = page.get_text("text")

            if DirectTextExtractor._is_toc_page(basic_text) or DirectTextExtractor._is_abbreviation_page(basic_text):
                # Use layout-preserving extraction for TOC and abbreviation pages
                text = DirectTextExtractor._extract_with_layout(page)
                # Light cleanup only - preserve structure
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = text.strip()
                page_type = "TOC" if DirectTextExtractor._is_toc_page(basic_text) else "abbreviations"
                logger.debug(f"Page {page_num + 1}: {page_type} detected, using layout extraction")
            else:
                # Standard extraction for normal pages
                text = basic_text
                # Standard cleanup
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r' {2,}', ' ', text)
                text = text.strip()

            page_texts.append(text)

            if progress_callback and (page_num + 1) % 50 == 0:
                progress_callback(OCRProgress(
                    current_page=page_num + 1,
                    total_pages=total_pages,
                    phase="extracting"
                ))

        doc.close()

        if progress_callback:
            progress_callback(OCRProgress(
                current_page=total_pages,
                total_pages=total_pages,
                phase="extracting"
            ))

        full_text = '\n\n'.join(page_texts)
        return full_text, total_pages, page_texts


# =============================================================================
# AZURE DOCUMENT INTELLIGENCE (FAST - For scanned PDFs)
# =============================================================================

class AzureOCRProvider:
    """
    Azure Document Intelligence - Fast cloud OCR.
    ~2-3 seconds per page, supports 300+ languages automatically.
    Cost: ~$1 per 100 pages ($6.37 for 637 pages)
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
    
    def is_available(self) -> bool:
        return bool(self.settings.azure_doc_endpoint and self.settings.azure_doc_key)
    
    def _get_client(self):
        if not self._client:
            try:
                from azure.ai.documentintelligence import DocumentIntelligenceClient
                from azure.core.credentials import AzureKeyCredential
                
                self._client = DocumentIntelligenceClient(
                    endpoint=self.settings.azure_doc_endpoint,
                    credential=AzureKeyCredential(self.settings.azure_doc_key)
                )
            except ImportError:
                logger.warning("Azure SDK not installed. Run: pip install azure-ai-documentintelligence")
                return None
        return self._client
    
    async def extract_text(
        self,
        file_path: str,
        progress_callback: ProgressCallback = None
    ) -> Tuple[str, int]:
        """Extract text using Azure Document Intelligence."""
        client = self._get_client()
        if not client:
            raise RuntimeError("Azure client not available")
        
        # Read file
        with open(file_path, 'rb') as f:
            document_bytes = f.read()
        
        # Process in executor to not block
        loop = asyncio.get_event_loop()
        
        def analyze():
            poller = client.begin_analyze_document(
                "prebuilt-read",
                document_bytes,
                content_type="application/octet-stream"
            )
            return poller.result()
        
        logger.info("Starting Azure OCR analysis...")
        result = await loop.run_in_executor(None, analyze)
        
        all_text = []
        page_count = len(result.pages) if result.pages else 0
        
        for i, page in enumerate(result.pages):
            page_text = []
            for line in page.lines:
                page_text.append(line.content)
            all_text.append('\n'.join(page_text))
            
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(OCRProgress(
                    current_page=i + 1,
                    total_pages=page_count,
                    phase="extracting"
                ))
        
        if progress_callback:
            progress_callback(OCRProgress(
                current_page=page_count,
                total_pages=page_count,
                phase="extracting"
            ))
        
        logger.info("Azure OCR complete", pages=page_count)
        return '\n\n'.join(all_text), page_count


# =============================================================================
# EASYOCR FALLBACK (SLOW - When no cloud APIs available)
# =============================================================================

class EasyOCRProvider:
    """
    EasyOCR - Free fallback option.
    Optimized with maximum parallelization but still slow on CPU.
    ~30-60 seconds per page on CPU.
    """
    
    # Maximum parallelization for speed
    BATCH_SIZE = 16
    MAX_WORKERS = 8
    DPI = 100  # Lower DPI for speed (still readable)
    
    def __init__(self):
        self._readers = {}  # Cache readers by language
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
    
    def _get_reader(self, languages: List[str]):
        """Get or create EasyOCR reader."""
        lang_key = tuple(sorted(languages))
        if lang_key not in self._readers:
            try:
                import easyocr
                logger.info("Loading EasyOCR for languages", languages=languages)
                self._readers[lang_key] = easyocr.Reader(
                    languages,
                    gpu=False,
                    verbose=False
                )
            except Exception as e:
                logger.error("EasyOCR init failed", error=str(e))
                raise
        return self._readers[lang_key]
    
    def is_available(self) -> bool:
        try:
            import easyocr
            return True
        except ImportError:
            return False
    
    async def extract_text(
        self,
        file_path: str,
        progress_callback: ProgressCallback = None,
        languages: List[str] = None
    ) -> Tuple[str, int]:
        """Extract text using EasyOCR with maximum parallelization."""
        languages = languages or ['uk', 'ru', 'en']
        reader = self._get_reader(languages)
        
        # Open PDF and render pages
        doc = fitz.open(file_path)
        page_count = len(doc)
        
        logger.info("EasyOCR: Processing pages", count=page_count)
        logger.warning("Estimated time", minutes_min=page_count * 30 // 60, minutes_max=page_count * 60 // 60)
        
        # Pre-render all pages at low DPI for speed
        page_images = []
        matrix = fitz.Matrix(self.DPI / 72, self.DPI / 72)
        
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=matrix)
            img_data = pix.tobytes("png")
            page_images.append((page_num, img_data))
        doc.close()
        
        # Process in parallel batches
        all_results = {}
        loop = asyncio.get_event_loop()
        processed = 0
        
        for batch_start in range(0, page_count, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, page_count)
            batch = page_images[batch_start:batch_end]
            
            futures = [
                loop.run_in_executor(
                    self._executor,
                    self._process_page,
                    reader,
                    img_data,
                    page_num
                )
                for page_num, img_data in batch
            ]
            
            batch_results = await asyncio.gather(*futures)
            
            for page_num, text in batch_results:
                all_results[page_num] = text
                processed += 1
            
            if progress_callback:
                progress_callback(OCRProgress(
                    current_page=processed,
                    total_pages=page_count,
                    phase="extracting"
                ))
            
            logger.info("EasyOCR progress", processed=processed, total=page_count)
        
        all_text = [all_results[i] for i in range(page_count)]
        return '\n\n'.join(all_text), page_count
    
    def _process_page(self, reader, img_data: bytes, page_num: int) -> Tuple[int, str]:
        """Process single page."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_data)
            tmp_path = tmp.name
        
        try:
            result = reader.readtext(tmp_path)
            text = ' '.join([det[1] for det in result if det[1]])
            return (page_num, text)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass


# =============================================================================
# GPT-4 VISION OCR (Premium alternative)
# =============================================================================

class GPT4OCRProvider:
    """
    GPT-4 Vision - Premium OCR using vision capabilities.
    Excellent for complex layouts and multilingual content.
    """
    
    BATCH_SIZE = 5  # API rate limit consideration
    
    def __init__(self):
        self.settings = get_settings()
    
    def is_available(self) -> bool:
        return bool(self.settings.openai_api_key)
    
    async def extract_text(
        self,
        file_path: str,
        progress_callback: ProgressCallback = None
    ) -> Tuple[str, int]:
        """Extract text using GPT-4 Vision."""
        import base64
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        file_path = Path(file_path)
        all_text = []
        page_count = 0
        
        prompt = """Extract ALL text from this document image accurately.
Preserve the original structure and formatting.
For any non-English text, extract it exactly as written (do not translate).
Return only the extracted text, nothing else."""
        
        if file_path.suffix.lower() == '.pdf':
            doc = fitz.open(str(file_path))
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode()
                
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    }],
                    max_tokens=4096
                )
                all_text.append(response.choices[0].message.content)
                
                if progress_callback:
                    progress_callback(OCRProgress(
                        current_page=page_num + 1,
                        total_pages=page_count,
                        phase="extracting"
                    ))
            
            doc.close()
        else:
            page_count = 1
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            img_base64 = base64.b64encode(img_bytes).decode()
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }],
                max_tokens=4096
            )
            all_text.append(response.choices[0].message.content)
        
        return '\n\n'.join(all_text), page_count


# =============================================================================
# MAIN OCR SERVICE - SMART HYBRID
# =============================================================================

class OCRService:
    """
    Smart Hybrid OCR Service
    ========================
    Automatically selects the fastest method for each document:
    
    1. Text-based PDFs → Direct extraction (instant)
    2. Scanned PDFs + Azure available → Azure OCR (fast)
    3. Scanned PDFs, no Azure → EasyOCR (slow fallback)
    
    Performance:
    - Direct: 637 pages in <10 seconds
    - Azure: 637 pages in 8-12 minutes
    - EasyOCR: 637 pages in ~5 hours (avoid if possible)
    """
    
    def __init__(self):
        self.analyzer = DocumentAnalyzer()
        self.azure = AzureOCRProvider()
        self.easyocr = EasyOCRProvider()
        self.gpt4 = GPT4OCRProvider()
    
    def get_available_providers(self) -> List[str]:
        """Get list of available OCR methods."""
        available = ['direct']  # Always available
        if self.azure.is_available():
            available.append('azure')
        if self.gpt4.is_available():
            available.append('gpt4o')
        if self.easyocr.is_available():
            available.append('easyocr')
        return available
    
    async def extract_text(
        self,
        file_path: str,
        provider: Optional[OCRProvider] = None,
        progress_callback: ProgressCallback = None,
        languages: List[str] = None
    ) -> Tuple[str, int, str, List[str]]:
        """
        Extract text from document using the optimal method.
        
        Returns:
            (extracted_text, page_count, method_used, page_texts_list)
        """
        file_path = str(file_path)
        
        # ALWAYS analyze first to check for embedded text
        logger.info("Analyzing document...")
        analysis = self.analyzer.analyze_pdf(file_path)
        
        logger.info(
            "Document analysis complete",
            pages=analysis['total_pages'],
            has_text=analysis['has_text'],
            text_coverage=f"{analysis['text_coverage']:.1%}",
            scripts=analysis['detected_scripts']
        )
        
        # PRIORITY: If PDF has embedded text, ALWAYS use direct extraction (instant)
        if analysis['has_text']:
            logger.info("Using DIRECT extraction (PDF has embedded text - this is fast!)")
            text, pages, page_texts = await DirectTextExtractor.extract(file_path, progress_callback)
            return text, pages, 'direct', page_texts
        
        # Only use OCR if PDF does NOT have embedded text (scanned document)
        logger.warning("PDF has no embedded text - must use OCR (slower)")
        
        # If provider explicitly set, use it
        if provider is not None:
            if provider == OCRProvider.AZURE and self.azure.is_available():
                text, pages = await self.azure.extract_text(file_path, progress_callback)
                # Convert to page list (Azure doesn't return per-page)
                page_texts = [text]  # Fallback: treat as single page
                return text, pages, 'azure', page_texts
            
            elif provider == OCRProvider.GPT4O and self.gpt4.is_available():
                text, pages = await self.gpt4.extract_text(file_path, progress_callback)
                page_texts = [text]
                return text, pages, 'gpt4o', page_texts
        
        # Auto-select OCR provider
        if self.azure.is_available():
            text, pages = await self.azure.extract_text(file_path, progress_callback)
            page_texts = [text]
            return text, pages, 'azure', page_texts
        
        if self.easyocr.is_available():
            if not languages:
                scripts = analysis.get('detected_scripts', set())
                languages = self._scripts_to_languages(scripts)
            text, pages = await self.easyocr.extract_text(file_path, progress_callback, languages)
            page_texts = [text]
            return text, pages, 'easyocr', page_texts
        
        raise RuntimeError("No text extraction method available")
    
    def _scripts_to_languages(self, scripts: set) -> List[str]:
        """Convert detected scripts to EasyOCR language codes."""
        languages = ['en']  # Always include English
        
        if 'cyrillic' in scripts:
            languages.extend(['uk', 'ru'])
        if 'arabic' in scripts:
            languages.append('ar')
        if 'chinese' in scripts:
            languages.append('ch_sim')
        if 'japanese' in scripts:
            languages.append('ja')
        if 'korean' in scripts:
            languages.append('ko')
        
        return list(set(languages))  # Deduplicate


# =============================================================================
# IMAGE EXTRACTION (NEW - For Word document with images)
# =============================================================================

class ImageExtractor:
    """
    Extracts images from PDF pages for inclusion in Word documents.
    """
    
    @staticmethod
    async def extract_images_from_pdf(file_path: str) -> List[List[dict]]:
        """
        Extract images from each page of a PDF.
        
        Returns:
            List of lists - each inner list contains image dicts for that page:
            [
                [  # Page 0
                    {"data": bytes, "ext": "png", "width": 800, "height": 600},
                    ...
                ],
                [  # Page 1
                    ...
                ],
                ...
            ]
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            ImageExtractor._extract_images_sync,
            file_path
        )
    
    @staticmethod
    def _extract_images_sync(file_path: str) -> List[List[dict]]:
        """Synchronous image extraction with position data."""
        all_page_images = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page_images = []
                page = doc.load_page(page_num)
                page_height = page.rect.height
                
                # Get list of images on this page
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]  # Image XREF
                        
                        # Get image position on page
                        img_rects = page.get_image_rects(xref)
                        y_position = 0
                        position = "bottom"  # default
                        
                        if img_rects:
                            y_position = img_rects[0].y0
                            y_percent = (y_position / page_height) * 100 if page_height > 0 else 50
                            if y_percent < 33:
                                position = "top"
                            elif y_percent < 66:
                                position = "middle"
                            else:
                                position = "bottom"
                        
                        # Extract image
                        base_image = doc.extract_image(xref)
                        
                        if base_image:
                            image_data = {
                                "data": base_image["image"],  # Raw bytes
                                "ext": base_image["ext"],      # Extension (png, jpeg, etc.)
                                "width": base_image.get("width", 0),
                                "height": base_image.get("height", 0),
                                "colorspace": base_image.get("colorspace", ""),
                                "y_position": y_position,      # Y coordinate on page
                                "position": position,          # "top", "middle", or "bottom"
                            }
                            page_images.append(image_data)
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                        continue
                
                # Sort images by Y position (top to bottom)
                page_images.sort(key=lambda x: x.get("y_position", 0))
                all_page_images.append(page_images)
            
            doc.close()
            logger.info(f"Extracted images from {len(all_page_images)} pages")
            
            # Log summary
            total_images = sum(len(imgs) for imgs in all_page_images)
            pages_with_images = sum(1 for imgs in all_page_images if imgs)
            logger.info(f"Total images: {total_images}, Pages with images: {pages_with_images}")
            
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            # Return empty lists for each page on failure
            return []
        
        return all_page_images
    
    @staticmethod
    def get_image_caption_pattern():
        """
        Returns regex patterns for detecting figure captions in multiple languages.
        These will be translated by the existing translation service.
        """
        return [
            r'Рис\.\s*\d+',      # Ukrainian: Рис. 1
            r'Рисунок\s*\d+',   # Ukrainian: Рисунок 1
            r'Fig\.\s*\d+',      # English: Fig. 1
            r'Figure\s*\d+',     # English: Figure 1
            r'Мал\.\s*\d+',      # Ukrainian: Мал. 1
            r'Малюнок\s*\d+',   # Ukrainian: Малюнок 1
            r'Табл\.\s*\d+',     # Table
            r'Таблиця\s*\d+',   # Ukrainian: Table
        ]
