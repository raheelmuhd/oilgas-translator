"""
Job Processor - Handles background document processing.
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from app.config import OCRProvider, TranslationProvider, get_settings
from app.models import JobStatus
from app.services.ocr_service import OCRService
from app.services.translation_service import TranslationService

logger = structlog.get_logger()


class JobProcessor:
    """Processes translation jobs in the background."""
    
    def __init__(self, db_session=None):
        self.settings = get_settings()
        self.ocr_service = OCRService()
        self.translation_service = TranslationService()
        self.db = db_session
    
    async def process_job(
        self,
        job_id: str,
        input_path: str,
        source_lang: Optional[str],
        target_lang: str,
        ocr_provider: Optional[str] = None,
        translation_provider: Optional[str] = None,
        status_callback=None
    ) -> dict:
        """
        Process a translation job.
        
        Steps:
        1. Extract text using OCR
        2. Detect language if not specified
        3. Split into chunks
        4. Translate chunks
        5. Save output
        """
        try:
            # Update status: Extracting
            if status_callback:
                await status_callback(
                    job_id, 
                    JobStatus.EXTRACTING, 
                    0, 
                    "Extracting text from document..."
                )
            
            # Step 1: OCR
            ocr_prov = OCRProvider(ocr_provider) if ocr_provider else None
            extracted_text, page_count, ocr_used = await self.ocr_service.extract_text(
                input_path, ocr_prov
            )
            
            if not extracted_text.strip():
                raise ValueError("No text could be extracted from the document")
            
            logger.info("Text extracted", pages=page_count, chars=len(extracted_text))
            
            # Step 2: Detect language if not specified
            if not source_lang:
                source_lang = await self.translation_service.detect_language(extracted_text)
                logger.info("Language detected", language=source_lang)
            
            # Step 3: Split into chunks
            chunks = self._split_into_chunks(extracted_text)
            total_chunks = len(chunks)
            
            if status_callback:
                await status_callback(
                    job_id,
                    JobStatus.TRANSLATING,
                    10,
                    f"Translating {total_chunks} sections...",
                    {"total_chunks": total_chunks, "total_pages": page_count}
                )
            
            # Step 4: Translate chunks
            trans_prov = TranslationProvider(translation_provider) if translation_provider else None
            translated_chunks = []
            
            for i, chunk in enumerate(chunks):
                translated, provider_used = await self.translation_service.translate(
                    chunk, source_lang, target_lang, trans_prov
                )
                translated_chunks.append(translated)
                
                # Update progress
                progress = 10 + (80 * (i + 1) / total_chunks)
                if status_callback:
                    await status_callback(
                        job_id,
                        JobStatus.TRANSLATING,
                        progress,
                        f"Translating section {i + 1}/{total_chunks}...",
                        {"processed_chunks": i + 1}
                    )
            
            # Step 5: Combine and save output
            translated_text = "\n\n".join(translated_chunks)
            
            # Generate output file
            output_filename = self._generate_output_filename(input_path, target_lang)
            output_path = Path(self.settings.output_dir) / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            
            if status_callback:
                await status_callback(
                    job_id,
                    JobStatus.COMPLETED,
                    100,
                    "Translation complete!",
                    {"output_path": str(output_path)}
                )
            
            return {
                "success": True,
                "output_path": str(output_path),
                "page_count": page_count,
                "chunk_count": total_chunks,
                "source_language": source_lang,
                "target_language": target_lang,
                "ocr_provider": ocr_used,
                "translation_provider": provider_used
            }
            
        except Exception as e:
            logger.error("Job processing failed", job_id=job_id, error=str(e))
            if status_callback:
                await status_callback(
                    job_id,
                    JobStatus.FAILED,
                    0,
                    f"Translation failed: {str(e)}"
                )
            raise
    
    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into manageable chunks for translation."""
        chunk_size = self.settings.chunk_size
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            if current_size + para_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def _generate_output_filename(self, input_path: str, target_lang: str) -> str:
        """Generate output filename."""
        input_name = Path(input_path).stem
        return f"{input_name}_translated_{target_lang}.txt"

