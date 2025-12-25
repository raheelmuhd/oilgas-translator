"""
Glossary Service - Oil & Gas terminology management.
Ensures technical terms are translated correctly.
"""

import json
import re
from pathlib import Path
from typing import Optional

import structlog

from app.config import get_settings

logger = structlog.get_logger()


class GlossaryService:
    """Manages oil & gas terminology glossary."""
    
    _instance = None
    _glossary = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if GlossaryService._glossary is None:
            self._load_glossary()
    
    def _load_glossary(self):
        """Load glossary from JSON file."""
        settings = get_settings()
        glossary_path = Path(settings.glossary_path)
        
        if glossary_path.exists():
            try:
                with open(glossary_path, 'r', encoding='utf-8') as f:
                    GlossaryService._glossary = json.load(f)
                logger.info("Glossary loaded", terms=len(GlossaryService._glossary.get('terms', [])))
            except Exception as e:
                logger.error("Failed to load glossary", error=str(e))
                GlossaryService._glossary = {"terms": [], "categories": {}}
        else:
            logger.warning("Glossary file not found", path=str(glossary_path))
            GlossaryService._glossary = {"terms": [], "categories": {}}
    
    def get_term(self, term: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Get translation for a specific term."""
        if not GlossaryService._glossary:
            return None
        
        for entry in GlossaryService._glossary.get("terms", []):
            # Check if term matches any language variant
            for lang, value in entry.get("translations", {}).items():
                if lang == source_lang and value.lower() == term.lower():
                    return entry["translations"].get(target_lang)
        
        return None
    
    def apply_terms(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Apply glossary terms to translated text.
        Ensures technical terms are correctly translated.
        """
        if not GlossaryService._glossary:
            return text
        
        result = text
        
        for entry in GlossaryService._glossary.get("terms", []):
            source_term = entry.get("translations", {}).get(source_lang)
            target_term = entry.get("translations", {}).get(target_lang)
            
            if source_term and target_term:
                # Create case-insensitive pattern
                pattern = re.compile(re.escape(source_term), re.IGNORECASE)
                result = pattern.sub(target_term, result)
        
        return result
    
    def get_context_for_prompt(self, source_lang: str, target_lang: str) -> str:
        """Generate glossary context for LLM prompts."""
        if not GlossaryService._glossary:
            return ""
        
        terms = []
        for entry in GlossaryService._glossary.get("terms", [])[:50]:  # Limit to 50 terms
            source_term = entry.get("translations", {}).get(source_lang)
            target_term = entry.get("translations", {}).get(target_lang)
            
            if source_term and target_term:
                terms.append(f"- {source_term} → {target_term}")
        
        if not terms:
            return ""
        
        return f"""OIL & GAS TERMINOLOGY GLOSSARY:
Use these translations for technical terms:
{chr(10).join(terms)}
"""
    
    def get_all_terms(self) -> list[dict]:
        """Get all glossary terms."""
        if not GlossaryService._glossary:
            return []
        return GlossaryService._glossary.get("terms", [])
    
    def get_categories(self) -> dict:
        """Get term categories."""
        if not GlossaryService._glossary:
            return {}
        return GlossaryService._glossary.get("categories", {})

