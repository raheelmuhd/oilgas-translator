"""
Translation Service - Handles text translation with multiple providers.
Supports: NLLB (free/CPU), Ollama (free/GPU), DeepSeek (budget), Claude (premium)
"""

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Optional

import httpx
import structlog

from app.config import TranslationProvider, get_settings
from app.models import NLLB_LANGUAGE_CODES
from app.services.glossary_service import GlossaryService

logger = structlog.get_logger()


class BaseTranslator(ABC):
    """Abstract base class for translation providers."""
    
    def __init__(self):
        self.glossary = GlossaryService()
    
    @abstractmethod
    async def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """Translate text from source to target language."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this translation provider is available."""
        pass
    
    def apply_glossary(self, text: str, source_lang: str, target_lang: str) -> str:
        """Apply oil & gas terminology glossary to translation."""
        return self.glossary.apply_terms(text, source_lang, target_lang)


class NLLBTranslator(BaseTranslator):
    """
    NLLB-200 (No Language Left Behind) - Free, runs on CPU.
    Meta's specialized translation model supporting 200+ languages.
    Accuracy: 85-90% for technical content.
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self._model = None
        self._tokenizer = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of NLLB model."""
        if not self._initialized:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                import torch
                
                logger.info("Loading NLLB model...", model=self.settings.nllb_model)
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.nllb_model
                )
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.settings.nllb_model
                )
                
                # Move to appropriate device
                device = self.settings.nllb_device
                if device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.cuda()
                
                self._model.eval()
                self._initialized = True
                logger.info("NLLB model loaded successfully")
                
            except Exception as e:
                logger.error("Failed to initialize NLLB", error=str(e))
                self._initialized = False
    
    def is_available(self) -> bool:
        return True  # NLLB is always available (will download if needed)
    
    async def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """Translate using NLLB model."""
        self._initialize()
        if not self._model:
            raise RuntimeError("NLLB model not initialized")
        
        # Convert language codes to NLLB format
        src_code = NLLB_LANGUAGE_CODES.get(source_lang, "eng_Latn")
        tgt_code = NLLB_LANGUAGE_CODES.get(target_lang, "eng_Latn")
        
        # Set source language
        self._tokenizer.src_lang = src_code
        
        # Process in chunks for long text
        max_length = 512
        chunks = self._split_text(text, max_length)
        translated_chunks = []
        
        for chunk in chunks:
            translated = await asyncio.get_event_loop().run_in_executor(
                None,
                self._translate_chunk,
                chunk,
                tgt_code
            )
            translated_chunks.append(translated)
        
        result = ' '.join(translated_chunks)
        
        # Apply glossary for technical terms
        result = self.apply_glossary(result, source_lang, target_lang)
        
        return result
    
    def _translate_chunk(self, text: str, target_lang: str) -> str:
        """Translate a single chunk of text."""
        import torch
        
        inputs = self._tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        if self.settings.nllb_device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(target_lang),
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        return self._tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def _split_text(self, text: str, max_length: int) -> list[str]:
        """Split text into chunks at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class OllamaTranslator(BaseTranslator):
    """
    Ollama - Free, runs locally. Best with GPU.
    Uses Llama or other models for translation.
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import httpx
            response = httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    async def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """Translate using Ollama."""
        from app.models import SUPPORTED_LANGUAGES
        
        src_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        tgt_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
        
        prompt = f"""Translate the following text from {src_name} to {tgt_name}. 
This is a technical oil and gas document. Maintain technical terminology accuracy.
Only output the translation, nothing else.

Text to translate:
{text}"""
        
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={
                    "model": self.settings.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()["response"]
        
        return self.apply_glossary(result, source_lang, target_lang)


class DeepSeekTranslator(BaseTranslator):
    """
    DeepSeek - Budget option, good quality.
    Cost: ~$0.28/$0.42 per million tokens.
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    def is_available(self) -> bool:
        return bool(self.settings.deepseek_api_key)
    
    async def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """Translate using DeepSeek API."""
        from app.models import SUPPORTED_LANGUAGES
        
        src_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        tgt_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
        
        system_prompt = """You are an expert translator specializing in oil and gas industry documents. 
Translate accurately while preserving technical terminology. 
Only output the translation, no explanations or notes."""
        
        user_prompt = f"Translate this from {src_name} to {tgt_name}:\n\n{text}"
        
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.settings.deepseek_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.settings.deepseek_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3
                }
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
        
        return self.apply_glossary(result, source_lang, target_lang)


class ClaudeTranslator(BaseTranslator):
    """
    Claude - Premium option, highest quality (78% "good" rating).
    Cost: ~$3/$15 per million tokens.
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
    
    def is_available(self) -> bool:
        return bool(self.settings.anthropic_api_key)
    
    async def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """Translate using Claude API."""
        from anthropic import AsyncAnthropic
        from app.models import SUPPORTED_LANGUAGES
        
        client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        
        src_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        tgt_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
        
        # Load glossary context
        glossary_context = self.glossary.get_context_for_prompt(source_lang, target_lang)
        
        system_prompt = f"""You are an expert translator specializing in oil and gas industry documents.

TRANSLATION GUIDELINES:
1. Translate from {src_name} to {tgt_name} with high accuracy
2. Preserve technical terminology exactly as used in the industry
3. Maintain document structure and formatting
4. Keep numbers, measurements, and units consistent
5. Translate abbreviations when appropriate, keep standard ones (API, BOP, etc.)

{glossary_context}

Only output the translation. No explanations, notes, or commentary."""

        response = await client.messages.create(
            model=self.settings.claude_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": f"Translate this text:\n\n{text}"}
            ],
            system=system_prompt
        )
        
        result = response.content[0].text
        return self.apply_glossary(result, source_lang, target_lang)


class TranslationService:
    """Main translation service that manages different providers."""
    
    def __init__(self):
        self.providers = {
            TranslationProvider.NLLB: NLLBTranslator(),
            TranslationProvider.OLLAMA: OllamaTranslator(),
            TranslationProvider.DEEPSEEK: DeepSeekTranslator(),
            TranslationProvider.CLAUDE: ClaudeTranslator(),
        }
    
    def get_available_providers(self) -> list[str]:
        """Get list of available translation providers."""
        return [p.value for p, impl in self.providers.items() if impl.is_available()]
    
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        provider: Optional[TranslationProvider] = None
    ) -> tuple[str, str]:
        """
        Translate text.
        Returns: (translated_text, provider_used)
        """
        settings = get_settings()
        
        # Select provider based on mode if not specified
        if provider is None:
            mode = settings.translation_mode.value
            if mode == "premium" and self.providers[TranslationProvider.CLAUDE].is_available():
                provider = TranslationProvider.CLAUDE
            elif mode == "budget" and self.providers[TranslationProvider.DEEPSEEK].is_available():
                provider = TranslationProvider.DEEPSEEK
            else:
                provider = TranslationProvider.NLLB
        
        translator = self.providers.get(provider)
        if not translator or not translator.is_available():
            # Fallback chain: Claude -> DeepSeek -> Ollama -> NLLB
            for fallback in [TranslationProvider.CLAUDE, TranslationProvider.DEEPSEEK, 
                           TranslationProvider.OLLAMA, TranslationProvider.NLLB]:
                if self.providers[fallback].is_available():
                    translator = self.providers[fallback]
                    provider = fallback
                    break
        
        logger.info("Starting translation", provider=provider.value, chars=len(text))
        translated = await translator.translate(text, source_lang, target_lang)
        logger.info("Translation complete", chars=len(translated))
        
        return translated, provider.value
    
    async def translate_chunks(
        self,
        chunks: list[str],
        source_lang: str,
        target_lang: str,
        provider: Optional[TranslationProvider] = None,
        progress_callback=None
    ) -> list[str]:
        """
        Translate multiple chunks with progress tracking.
        """
        translated = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            result, _ = await self.translate(chunk, source_lang, target_lang, provider)
            translated.append(result)
            
            if progress_callback:
                progress_callback((i + 1) / total * 100)
        
        return translated
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            # Use a simple heuristic or langdetect
            from langdetect import detect
            lang = detect(text[:1000])  # Use first 1000 chars
            return lang
        except:
            return "en"  # Default to English

