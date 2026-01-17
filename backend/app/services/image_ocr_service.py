"""
Cascading Image OCR Service (Generalized)
==========================================
Extracts text from PDF images using a cascading approach for best accuracy:

1. Tesseract (fastest, good for clean typed text)
2. PaddleOCR (if Tesseract confidence < 90%)
3. Ollama Vision minicpm-v (if still < 90%, best for handwritten/complex)

After extraction, translates using qwen3:8b via Ollama.
"""

import asyncio
import base64
import io
import os
import re
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

import httpx
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# =============================================================================
# WINDOWS TESSERACT CONFIGURATION
# =============================================================================
# Configure Tesseract path for Windows before importing pytesseract
if sys.platform == 'win32':
    TESSERACT_PATHS = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.expandvars(r'%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe'),
    ]
    for path in TESSERACT_PATHS:
        if os.path.exists(path):
            os.environ['TESSERACT_CMD'] = path
            break

# Confidence threshold to try next backend (90% = high quality requirement)
CONFIDENCE_THRESHOLD = 0.90


@dataclass
class ImageOCRResult:
    """Result from OCR operation on an image."""
    original_text: str
    translated_text: str
    confidence: float
    backend_used: str
    language_detected: str = ""


class ImagePreprocessor:
    """Preprocesses images to improve OCR accuracy."""
    
    @staticmethod
    def enhance_for_ocr(image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy."""
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        gray = image.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)
        gray = gray.filter(ImageFilter.SHARPEN)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
        
        return gray
    
    @staticmethod
    def resize_for_ocr(image: Image.Image, min_width: int = 1000) -> Image.Image:
        """Resize image to optimal size for OCR."""
        if image.width < min_width:
            scale = min_width / image.width
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image


# =============================================================================
# LANGUAGE MAPPING
# =============================================================================
TESSERACT_LANG_MAP = {
    'uk': 'ukr', 'ru': 'rus', 'hr': 'hrv', 'sr': 'srp', 'sl': 'slv',
    'pl': 'pol', 'cs': 'ces', 'sk': 'slk', 'bg': 'bul', 'mk': 'mkd',
    'be': 'bel', 'bs': 'bos',
    'en': 'eng', 'de': 'deu', 'fr': 'fra', 'es': 'spa', 'it': 'ita',
    'pt': 'por', 'nl': 'nld', 'da': 'dan', 'sv': 'swe', 'no': 'nor',
    'fi': 'fin', 'is': 'isl',
    'ro': 'ron', 'hu': 'hun', 'et': 'est', 'lv': 'lav', 'lt': 'lit',
    'zh': 'chi_sim', 'ja': 'jpn', 'ko': 'kor', 'vi': 'vie', 'th': 'tha',
    'ar': 'ara', 'he': 'heb', 'hi': 'hin', 'tr': 'tur', 'el': 'ell',
}

PADDLE_LANG_MAP = {
    'en': 'en', 'de': 'german', 'fr': 'french', 'es': 'spanish',
    'it': 'italian', 'pt': 'pt', 'nl': 'nl', 'pl': 'polish',
    'uk': 'cyrillic', 'ru': 'cyrillic', 'bg': 'cyrillic', 'sr': 'cyrillic',
    'be': 'cyrillic', 'mk': 'cyrillic',
    'hr': 'latin', 'sl': 'latin', 'cs': 'latin', 'sk': 'latin', 'bs': 'latin',
    'zh': 'ch', 'ja': 'japan', 'ko': 'korean', 'ar': 'ar',
    'default': 'en'
}

LANGUAGE_NAMES = {
    'uk': 'Ukrainian', 'ru': 'Russian', 'hr': 'Croatian', 'sr': 'Serbian',
    'sl': 'Slovenian', 'pl': 'Polish', 'cs': 'Czech', 'sk': 'Slovak',
    'bg': 'Bulgarian', 'en': 'English', 'de': 'German', 'fr': 'French',
    'es': 'Spanish', 'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch',
    'ro': 'Romanian', 'hu': 'Hungarian', 'zh': 'Chinese', 'ja': 'Japanese',
    'ko': 'Korean', 'ar': 'Arabic', 'tr': 'Turkish', 'el': 'Greek',
}


def clean_markdown(text: str) -> str:
    """Remove markdown formatting from text."""
    if not text:
        return ""
    
    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Clean whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# =============================================================================
# BACKEND 1: TESSERACT (Fastest)
# =============================================================================
class TesseractOCR:
    """Tesseract OCR - fast, good for clean typed documents."""
    
    def __init__(self):
        self._available = None
        self._installed_languages = None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            import pytesseract
            
            # Set Tesseract path on Windows
            if sys.platform == 'win32':
                tesseract_cmd = os.environ.get('TESSERACT_CMD')
                if tesseract_cmd and os.path.exists(tesseract_cmd):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                    logger.info(f"Tesseract path: {tesseract_cmd}")
            
            pytesseract.get_tesseract_version()
            self._installed_languages = pytesseract.get_languages()
            self._available = True
            logger.info(f"Tesseract available: {self._installed_languages}")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self._available = False
        
        return self._available
    
    def get_lang_string(self, source_lang: str) -> str:
        if not self._installed_languages:
            return 'eng'
        
        langs_to_use = []
        if source_lang:
            tess_lang = TESSERACT_LANG_MAP.get(source_lang, source_lang)
            if tess_lang in self._installed_languages:
                langs_to_use.append(tess_lang)
        
        if 'eng' in self._installed_languages and 'eng' not in langs_to_use:
            langs_to_use.append('eng')
        
        return '+'.join(langs_to_use) if langs_to_use else 'eng'
    
    def extract_text(self, image: Image.Image, source_lang: str = None) -> Tuple[str, float]:
        import pytesseract
        
        lang_str = self.get_lang_string(source_lang)
        
        try:
            data = pytesseract.image_to_data(
                image, lang=lang_str, output_type=pytesseract.Output.DICT
            )
            
            texts = []
            confidences = []
            
            for i, text in enumerate(data['text']):
                if text and str(text).strip():
                    texts.append(str(text).strip())
                    conf = int(data['conf'][i])
                    if conf > 0:
                        confidences.append(conf)
            
            full_text = ' '.join(texts)
            avg_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
            
            return full_text, avg_conf
            
        except Exception as e:
            logger.error(f"Tesseract failed: {e}")
            return "", 0.0


# =============================================================================
# BACKEND 2: PADDLEOCR
# =============================================================================
class PaddleOCRBackend:
    """PaddleOCR - better for tables and complex layouts."""
    
    def __init__(self):
        self._ocr_instances = {}
        self._available = None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            # Skip the slow connectivity check in PaddleOCR 3.0+
            os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
            from paddleocr import PaddleOCR
            self._available = True
            logger.info("PaddleOCR available")
        except ImportError:
            self._available = False
            logger.warning("PaddleOCR not installed")
        except Exception as e:
            self._available = False
            logger.warning(f"PaddleOCR error: {e}")
        
        return self._available
    
    def _get_ocr(self, source_lang: str = None):
        paddle_lang = PADDLE_LANG_MAP.get(source_lang, 'en')
        
        if paddle_lang not in self._ocr_instances:
            try:
                from paddleocr import PaddleOCR
                # PaddleOCR 3.0+ API - no use_gpu parameter (auto-detects GPU)
                self._ocr_instances[paddle_lang] = PaddleOCR(
                    lang=paddle_lang, use_angle_cls=False
                )
                logger.info(f"PaddleOCR initialized: lang={paddle_lang}")
            except Exception as e:
                logger.error(f"PaddleOCR init failed: {e}")
                return None
        
        return self._ocr_instances.get(paddle_lang)
    
    def extract_text(self, image: Image.Image, source_lang: str = None) -> Tuple[str, float]:
        try:
            import numpy as np
            
            ocr = self._get_ocr(source_lang)
            if ocr is None:
                return "", 0.0
            
            img_array = np.array(image.convert('RGB'))
            result = ocr.ocr(img_array, cls=False)
            
            texts = []
            confidences = []
            
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        text_conf = line[1]
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                            text, conf = str(text_conf[0]), float(text_conf[1])
                        else:
                            text, conf = str(text_conf), 0.8
                        
                        if text.strip():
                            texts.append(text.strip())
                            confidences.append(conf)
            
            full_text = ' '.join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            return full_text, avg_conf
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return "", 0.0


# =============================================================================
# BACKEND 3: OLLAMA VISION (Best accuracy)
# =============================================================================
class OllamaVisionOCR:
    """Ollama Vision OCR using minicpm-v."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "minicpm-v"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._available = None
    
    async def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = [m.get('name', '').split(':')[0] for m in response.json().get('models', [])]
                    self._available = self.model in models or any(self.model in m for m in models)
                    if self._available:
                        logger.info(f"Ollama Vision available: {self.model}")
                    return self._available
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
        
        self._available = False
        return False
    
    def _image_to_base64(self, image: Image.Image) -> str:
        max_dim = 1500  # Reduced for speed
        if image.width > max_dim or image.height > max_dim:
            ratio = min(max_dim / image.width, max_dim / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)  # JPEG is faster
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def extract_text(self, image: Image.Image, source_lang: str = None) -> Tuple[str, float]:
        """Extract text using Ollama Vision - returns clean text without markdown."""
        image_b64 = self._image_to_base64(image)
        
        lang_name = LANGUAGE_NAMES.get(source_lang, source_lang) if source_lang else None
        lang_hint = f"Document is in {lang_name}. " if lang_name else ""
        
        prompt = f"""{lang_hint}Extract ALL text from this image exactly as written.
Output plain text only. No markdown, no formatting, no asterisks.
Include numbers, dates, table data. Do NOT translate."""

        try:
            async with httpx.AsyncClient(timeout=90) as client:  # Increased timeout
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": [image_b64],
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 2048}  # Reduced tokens
                    }
                )
                
                if response.status_code == 200:
                    text = response.json().get('response', '').strip()
                    text = clean_markdown(text)  # Remove any markdown
                    logger.info(f"Ollama Vision: {len(text)} chars")
                    return text, 0.95
                else:
                    logger.error(f"Ollama Vision HTTP error: {response.status_code} - {response.text[:200]}")
                    
        except httpx.TimeoutException as e:
            logger.error(f"Ollama Vision timeout: {e}")
        except Exception as e:
            logger.error(f"Ollama Vision failed: {type(e).__name__}: {e}")
        
        return "", 0.0


# =============================================================================
# OLLAMA TRANSLATOR (Using qwen3:8b)
# =============================================================================
class OllamaTranslator:
    """Translates text using qwen3:8b via Ollama."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    async def translate(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        """Translate text using qwen3:8b."""
        if not text or not text.strip():
            return ""
        
        if source_lang == target_lang:
            return text
        
        source_name = LANGUAGE_NAMES.get(source_lang, source_lang or "the original language")
        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        
        prompt = f"""/no_think
Translate this {source_name} text to {target_name}. Keep formatting. Only output the translation:

{text}"""

        try:
            async with httpx.AsyncClient(timeout=45) as client:  # Fast timeout
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 2048}
                    }
                )
                
                if response.status_code == 200:
                    translated = response.json().get('response', '').strip()
                    logger.info(f"Translated: {len(text)} → {len(translated)} chars")
                    return translated
                    
        except Exception as e:
            logger.error(f"Translation failed: {e}")
        
        return text  # Return original if failed


# =============================================================================
# MAIN SERVICE
# =============================================================================
class ImageOCRService:
    """Cascading OCR with translation via qwen3:8b."""
    
    def __init__(self, translation_service=None, settings=None, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.translation_service = translation_service
        self.settings = settings
        self.confidence_threshold = confidence_threshold
        self.preprocessor = ImagePreprocessor()
        
        # Initialize OCR backends
        self.tesseract = TesseractOCR()
        self.paddleocr = PaddleOCRBackend()
        
        ollama_url = getattr(settings, 'ollama_base_url', 'http://localhost:11434') if settings else 'http://localhost:11434'
        self.ollama_vision = OllamaVisionOCR(base_url=ollama_url, model="minicpm-v")
        
        # Initialize translator
        self.translator = OllamaTranslator(base_url=ollama_url, model="qwen3:8b")
        
        backends = []
        if self.tesseract.is_available():
            backends.append("Tesseract")
        if self.paddleocr.is_available():
            backends.append("PaddleOCR")
        logger.info(f"OCR initialized: {backends} + Ollama Vision + qwen3:8b translator")
    
    def is_available(self) -> bool:
        return self.tesseract.is_available() or self.paddleocr.is_available()
    
    async def extract_text_cascading(self, image: Image.Image, source_lang: str = None) -> Tuple[str, float, str]:
        """Try OCR backends until good accuracy achieved."""
        img_enhanced = self.preprocessor.enhance_for_ocr(image)
        img_resized = self.preprocessor.resize_for_ocr(img_enhanced)
        
        # STEP 1: Tesseract (fastest)
        if self.tesseract.is_available():
            text, conf = self.tesseract.extract_text(img_resized, source_lang)
            logger.info(f"Tesseract: {len(text)} chars, {conf:.0%}")
            if conf >= self.confidence_threshold and len(text) > 20:
                return text, conf, "tesseract"
        
        # STEP 2: PaddleOCR
        if self.paddleocr.is_available():
            text, conf = self.paddleocr.extract_text(img_resized, source_lang)
            logger.info(f"PaddleOCR: {len(text)} chars, {conf:.0%}")
            if conf >= self.confidence_threshold and len(text) > 20:
                return text, conf, "paddleocr"
        
        # STEP 3: Ollama Vision (best but slower)
        if await self.ollama_vision.is_available():
            text, conf = await self.ollama_vision.extract_text(image, source_lang)
            logger.info(f"Ollama Vision: {len(text)} chars")
            if text:
                return text, conf, "ollama_vision"
        
        # Fallback
        if self.tesseract.is_available():
            text, conf = self.tesseract.extract_text(image, source_lang)
            if text:
                return text, conf, "tesseract_fallback"
        
        return "", 0.0, "none"
    
    async def extract_and_translate(self, image_data: bytes, source_lang: str = None, target_lang: str = "en") -> ImageOCRResult:
        """Extract text and translate using qwen3:8b."""
        try:
            img = Image.open(io.BytesIO(image_data))
            
            if img.width < 50 or img.height < 50:
                return ImageOCRResult("", "", 0.0, "skipped")
            
            # Extract text
            original_text, confidence, backend = await self.extract_text_cascading(img, source_lang)
            
            if not original_text:
                return ImageOCRResult("", "", 0.0, backend)
            
            # Translate using qwen3:8b directly
            translated_text = original_text
            if source_lang != target_lang:
                translated_text = await self.translator.translate(original_text, source_lang, target_lang)
            
            return ImageOCRResult(
                original_text=original_text,
                translated_text=translated_text,
                confidence=confidence,
                backend_used=backend,
                language_detected=source_lang or ""
            )
            
        except Exception as e:
            logger.error(f"OCR pipeline failed: {e}")
            return ImageOCRResult("", "", 0.0, "error")
    
    async def process_page_images(self, page_images: List[Dict], source_lang: str = None, target_lang: str = "en") -> List[Dict]:
        """Process all images from a page."""
        for img in page_images:
            img_data = img.get('data') or img.get('bytes')
            if not img_data:
                continue
            
            try:
                result = await self.extract_and_translate(img_data, source_lang, target_lang)
                
                img['ocr_text'] = result.original_text
                img['translated_text'] = result.translated_text
                img['ocr_confidence'] = result.confidence
                img['ocr_backend'] = result.backend_used
                
                if result.translated_text:
                    img['image_label'] = f"[IMAGE TEXT TRANSLATION]\n{result.translated_text}"
                    logger.info(f"OCR ({result.backend_used}): {len(result.original_text)}→{len(result.translated_text)} chars")
                else:
                    img['image_label'] = ""
                    
            except Exception as e:
                logger.warning(f"Image failed: {e}")
                img['ocr_text'] = ""
                img['translated_text'] = ""
                img['image_label'] = ""
        
        return page_images


async def process_all_images(
    all_page_images: List[List[Dict]],
    translation_service=None,
    source_lang: str = None,
    target_lang: str = "en",
    settings=None,
    progress_callback=None
) -> List[List[Dict]]:
    """Process images from all pages with cascading OCR + translation."""
    service = ImageOCRService(translation_service, settings)
    
    if not service.is_available():
        logger.warning("No OCR backend available")
        return all_page_images
    
    total_images = sum(len(imgs) for imgs in all_page_images)
    if total_images == 0:
        return all_page_images
    
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang) if source_lang else "auto"
    logger.info(f"Processing {total_images} images (lang={lang_name})")
    
    processed = []
    images_done = 0
    backend_stats = {"tesseract": 0, "paddleocr": 0, "ollama_vision": 0, "none": 0}
    
    for page_idx, page_images in enumerate(all_page_images):
        if page_images:
            processed_page = await service.process_page_images(page_images, source_lang, target_lang)
            processed.append(processed_page)
            
            for img in processed_page:
                backend = img.get('ocr_backend', 'none')
                if backend in backend_stats:
                    backend_stats[backend] += 1
                elif 'tesseract' in backend:
                    backend_stats['tesseract'] += 1
            
            images_done += len(page_images)
            if progress_callback:
                progress_callback(f"OCR: {images_done}/{total_images}")
        else:
            processed.append([])
    
    images_with_text = sum(1 for page in processed for img in page if img.get('translated_text'))
    logger.info(f"Complete: {images_with_text}/{total_images} images translated")
    logger.info(f"Backends: Tesseract={backend_stats['tesseract']}, Paddle={backend_stats['paddleocr']}, Ollama={backend_stats['ollama_vision']}")
    
    return processed
