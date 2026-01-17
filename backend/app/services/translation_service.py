"""
Translation Service - TOC DOT LEADER FIX
=========================================
Strips dot leaders before translation, restores after.
"""

import asyncio
import re
import os
from datetime import datetime
from typing import Optional, List, Tuple

import httpx
import structlog

from app.config import TranslationProvider, get_settings
from app.services.glossary_service import GlossaryService

logger = structlog.get_logger()

# Debug log file
DEBUG_LOG = os.path.join(os.path.expanduser("~"), "translation_debug.log")

def debug_log(msg: str, data: dict = None):
    """Write to debug log file."""
    timestamp = datetime.now().isoformat()
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{timestamp}] {msg}\n")
        if data:
            for k, v in data.items():
                if isinstance(v, str) and len(v) > 500:
                    f.write(f"  {k}: {v[:500]}... (truncated, {len(v)} chars)\n")
                else:
                    f.write(f"  {k}: {v}\n")


class BaseTranslator:
    def __init__(self):
        self.glossary = GlossaryService()


class OllamaTranslator(BaseTranslator):
    """Translation with smart TOC handling."""

    CHUNK_SIZE_CHARS = 800

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self._available = None
        # Clear debug log on init
        with open(DEBUG_LOG, "w") as f:
            f.write(f"=== Translation Debug Log Started {datetime.now()} ===\n")

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            response = httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Ollama API can return either:
                # - {"models": [...]} (dict format)
                # - [...] (list format directly)
                if isinstance(data, dict):
                    models = data.get("models", [])
                elif isinstance(data, list):
                    models = data
                else:
                    models = []
                
                # Extract model names - handle both dict and list formats
                model_names = []
                for m in models:
                    if isinstance(m, dict):
                        model_names.append(m.get("name", ""))
                    elif isinstance(m, str):
                        model_names.append(m)
                
                self._available = any(self.settings.ollama_model in name for name in model_names)
            else:
                self._available = False
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            self._available = False
        return self._available

    def _is_toc_content(self, text: str) -> bool:
        """Detect if text is table of contents (has dot leaders)."""
        # Count lines with dot leaders
        dot_leader_pattern = r'\.\s*\.\s*\.'
        lines_with_dots = len(re.findall(dot_leader_pattern, text))
        total_lines = len([l for l in text.split('\n') if l.strip()])
        
        # If more than 30% of lines have dot leaders, it's TOC
        if total_lines > 0 and lines_with_dots / total_lines > 0.3:
            return True
        return False

    def _is_abbreviation_content(self, text: str) -> bool:
        """Detect if text is an abbreviation list (has em-dashes)."""
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return False
        
        # Count lines with em-dash or regular dash pattern (ABBR — meaning)
        dash_pattern = r'^[А-ЯІЇЄҐA-Z]{2,10}\s*[—\-–]\s*.+'
        lines_with_dashes = sum(1 for l in lines if re.match(dash_pattern, l.strip()))
        
        # If more than 50% of lines are abbreviation format, it's abbreviations
        if lines_with_dashes / len(lines) > 0.5:
            return True
        return False

    def _parse_toc_entries(self, text: str) -> List[dict]:
        """Parse TOC into structured entries."""
        entries = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if not line.strip():
                entries.append({'type': 'empty', 'text': '', 'page': None})
                i += 1
                continue
            
            # Check if line has page number at end (with dots before it)
            match = re.match(r'^(.+?)\s*\.[\s.]*(\d+)\s*$', line)
            
            if match:
                title = match.group(1).strip()
                page = match.group(2)
                entries.append({'type': 'toc_entry', 'text': title, 'page': page})
            else:
                # Line without page number - might be continuation or header
                # Check if next line(s) complete this entry
                combined = line.strip()
                j = i + 1
                
                # Look ahead for continuation lines
                while j < len(lines):
                    next_line = lines[j]
                    next_match = re.match(r'^(.+?)\s*\.[\s.]*(\d+)\s*$', next_line)
                    
                    if next_match:
                        # This line completes the entry
                        combined += ' ' + next_match.group(1).strip()
                        page = next_match.group(2)
                        entries.append({'type': 'toc_entry', 'text': combined, 'page': page})
                        i = j
                        break
                    elif next_line.strip() and not re.match(r'^\d+\.', next_line):
                        # Continuation line (doesn't start with number)
                        combined += ' ' + next_line.strip()
                        j += 1
                    else:
                        # Not a continuation
                        entries.append({'type': 'header', 'text': combined, 'page': None})
                        i = j - 1
                        break
                else:
                    # End of text
                    entries.append({'type': 'header', 'text': combined, 'page': None})
            
            i += 1
        
        return entries

    def _format_toc_entry(self, text: str, page: str, total_width: int = 90) -> str:
        """Reconstruct TOC line with dot leaders."""
        if page is None:
            return text
        
        # Calculate dots needed
        dots_space = total_width - len(text) - len(page) - 2
        if dots_space < 5:
            dots_space = 5
        
        dots = ' ' + '.' * dots_space + ' '
        return f"{text}{dots}{page}"

    async def _translate_simple(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Simple translation without dots."""
        prompt = f"Translate to {tgt_lang}. Output ONLY the translation:\n\n{text}"
        
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json={
                        "model": self.settings.ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": 0.1,      # Lower = faster, still accurate
                            "num_predict": 1500,      # Reduced for speed (was 2000)
                            "num_ctx": 4096,
                        }
                    }
                )
                response.raise_for_status()
                result = response.json().get("message", {}).get("content", "")
                return self._clean_response(result)
        except Exception as e:
            debug_log(f"Translation error", {"error": str(e)})
            return text

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Main translation with TOC and abbreviation detection."""
        if not text.strip():
            return ""

        lang_names = {"uk": "Ukrainian", "en": "English", "ru": "Russian"}
        src_name = lang_names.get(source_lang, source_lang)
        tgt_name = lang_names.get(target_lang, target_lang)

        debug_log("=== NEW PAGE TRANSLATION ===", {
            "source_lang": src_name,
            "target_lang": tgt_name,
            "input_length": len(text),
            "is_toc": self._is_toc_content(text),
            "is_abbreviations": self._is_abbreviation_content(text)
        })

        # Check content type and use appropriate strategy
        if self._is_toc_content(text):
            debug_log("Detected TOC content - using smart TOC translation")
            return await self._translate_toc(text, src_name, tgt_name)
        elif self._is_abbreviation_content(text):
            debug_log("Detected abbreviation content - using abbreviation translation")
            return await self._translate_abbreviations(text, src_name, tgt_name)
        else:
            debug_log("Regular content - using standard translation")
            return await self._translate_narrative(text, src_name, tgt_name)

    async def _translate_toc(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Smart TOC translation - strip dots, translate, restore."""
        
        # Step 1: Parse TOC entries
        entries = self._parse_toc_entries(text)
        debug_log(f"Parsed {len(entries)} TOC entries")
        
        # Step 2: Extract texts to translate (skip empty lines)
        texts_to_translate = []
        entry_indices = []  # Track which entries need translation
        
        for i, entry in enumerate(entries):
            if entry['type'] != 'empty' and entry['text']:
                texts_to_translate.append(entry['text'])
                entry_indices.append(i)
        
        if not texts_to_translate:
            return text
        
        # Step 3: Translate all texts at once (batch for efficiency)
        combined = "\n".join(texts_to_translate)
        debug_log("Translating TOC texts (without dots)", {"text_count": len(texts_to_translate)})
        
        translated = await self._translate_simple(combined, src_lang, tgt_lang)
        
        if not translated:
            debug_log("TOC translation failed - returning original")
            return text
        
        translated_lines = translated.split('\n')
        debug_log(f"Got {len(translated_lines)} translated lines")
        
        # Step 4: Reconstruct with dot leaders
        result_lines = []
        trans_idx = 0
        
        for i, entry in enumerate(entries):
            if entry['type'] == 'empty':
                result_lines.append('')
            elif i in entry_indices and trans_idx < len(translated_lines):
                trans_text = translated_lines[trans_idx].strip()
                trans_idx += 1
                
                if entry['page']:
                    result_lines.append(self._format_toc_entry(trans_text, entry['page']))
                else:
                    result_lines.append(trans_text)
            else:
                # Fallback to original
                if entry['page']:
                    result_lines.append(self._format_toc_entry(entry['text'], entry['page']))
                else:
                    result_lines.append(entry['text'])
        
        result = '\n'.join(result_lines)
        debug_log("TOC translation complete", {"result_length": len(result)})
        return result

    async def _translate_abbreviations(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate abbreviation lists - use simple prompt that works."""
        
        # Simple prompt - this worked in testing!
        prompt = f"Translate to {tgt_lang}. Keep the same format (abbreviation — meaning). Transliterate Cyrillic abbreviations to Latin:\n\n{text}"
        
        debug_log("Translating abbreviations as single block", {"length": len(text)})
        
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json={
                        "model": self.settings.ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": 0.1,      # Lower = faster, still accurate
                            "num_predict": 1500,      # Reduced for speed (was 2000)
                            "num_ctx": 4096,
                        }
                    }
                )
                response.raise_for_status()
                result = response.json().get("message", {}).get("content", "")
                result = self._clean_response(result)
                
                if result:
                    debug_log("Abbreviation translation complete", {"result_length": len(result)})
                    return result
                else:
                    debug_log("Abbreviation translation empty - returning original")
                    return text
                    
        except Exception as e:
            debug_log(f"Abbreviation translation error", {"error": str(e)})
            return text

    async def _translate_narrative(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate narrative content with context chaining."""
        
        # Split into chunks
        chunks = self._split_into_chunks(text)
        debug_log(f"Split narrative into {len(chunks)} chunks")
        
        if len(chunks) == 1:
            return await self._translate_simple(chunks[0], src_lang, tgt_lang)
        
        # Translate with context
        translated_chunks = []
        prev_translation = ""
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                translated_chunks.append(chunk)
                continue
            
            if prev_translation:
                prompt = f"""Translate to {tgt_lang}. Keep same structure.

CONTEXT (previous translation for consistency):
{prev_translation[-300:]}

TRANSLATE THIS:
{chunk}

OUTPUT (translation only):"""
            else:
                prompt = f"Translate to {tgt_lang}. Output ONLY the translation:\n\n{chunk}"
            
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(
                        f"{self.settings.ollama_base_url}/api/chat",
                        json={
                            "model": self.settings.ollama_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {
                                "temperature": 0.2,
                                "num_predict": 2000,
                                "num_ctx": 4096,
                            }
                        }
                    )
                    result = response.json().get("message", {}).get("content", "")
                    result = self._clean_response(result)
                    
                    if result:
                        translated_chunks.append(result)
                        prev_translation = result
                    else:
                        translated_chunks.append(chunk)
                        prev_translation = chunk
                        
            except Exception as e:
                debug_log(f"Chunk {i} error", {"error": str(e)})
                translated_chunks.append(chunk)
        
        return '\n'.join(translated_chunks)

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks at paragraph boundaries."""
        if len(text) <= self.CHUNK_SIZE_CHARS:
            return [text]
        
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) > self.CHUNK_SIZE_CHARS and current:
                chunks.append(current.strip())
                current = para
            else:
                current = current + '\n\n' + para if current else para
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks if chunks else [text]

    def _clean_response(self, text: str) -> str:
        """Clean LLM response."""
        if not text:
            return ""
        
        # Remove think tags (both with content and standalone)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
        
        # Remove /think artifacts (Qwen3 sometimes outputs this)
        text = re.sub(r'\s*/think\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'/no\s*think', '', text, flags=re.IGNORECASE)
        
        text = text.strip()
        
        # Remove common prefixes
        prefixes = [
            "here is the translation:",
            "here's the translation:",
            "translation:",
            "output:",
        ]
        
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        return text


class TranslationService:
    """Translation service with TOC handling."""

    def __init__(self):
        self.settings = get_settings()
        self.ollama = OllamaTranslator()
        # For compatibility with main.py
        self.providers = self.get_available_providers()
        logger.info("TranslationService initialized with TOC fix")
        logger.info(f"Debug log: {DEBUG_LOG}")

    def get_available_providers(self) -> List[str]:
        providers = []
        if self.ollama.is_available():
            providers.append("ollama")
        return providers

    def needs_translation(self, text: str, target_lang: str = "en") -> bool:
        if not text or len(text.strip()) < 20:
            return False
        latin = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        non_latin = sum(1 for c in text if c.isalpha() and ord(c) >= 128)
        total = latin + non_latin
        if total == 0:
            return False
        if target_lang.lower() in ["en", "eng"] and (latin / total) > 0.7:
            return False
        return True

    async def translate_if_needed(
        self, text: str, source_lang: str, target_lang: str,
        provider: Optional[TranslationProvider] = None
    ) -> Tuple[str, str, bool]:
        if not self.needs_translation(text, target_lang):
            return text, "skipped", False
        translated, provider_used = await self.translate(text, source_lang, target_lang, provider)
        return translated, provider_used, True

    async def translate(
        self, text: str, source_lang: str, target_lang: str,
        provider: Optional[TranslationProvider] = None
    ) -> Tuple[str, str]:
        translated = await self.ollama.translate(text, source_lang, target_lang)
        return translated, "ollama"

    async def detect_language(self, text: str) -> str:
        try:
            from langdetect import detect
            return detect(text[:1000])
        except:
            cyrillic = sum(1 for c in text[:500] if '\u0400' <= c <= '\u04FF')
            return "uk" if cyrillic > 50 else "en"

    async def detect_language_details(self, text: str) -> dict:
        lang = await self.detect_language(text)
        return {"top": lang, "candidates": [{"lang": lang, "prob": None}]}
