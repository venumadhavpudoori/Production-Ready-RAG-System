"""LLM Answer Generator with Citation Grounding."""

import time
import logging
from typing import List, Tuple, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.rag.chunking import DocumentChunk
from src.rag.generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_context_passages
from src.rag.config import config

logger = logging.getLogger(__name__)


class CitationGenerator:
    """Generates answers with grounded inline citations using Gemini."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or config.generator_model_name
        self.api_key = api_key or config.gemini_api_key
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        if self.api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    max_retries=2,
                    timeout=20.0,
                )
                logger.info("Initialized Gemini LLM (%s)", self.model_name)
            except Exception as e:
                logger.warning("Failed to initialize ChatGoogleGenerativeAI (%s), fallback will be used", e)
                self.llm = None
        else:
            logger.info("No Gemini API key provided; fallback extractive generator active.")

    def _extract_text(self, content) -> str:
        """Extract clean string content from response regardless of format."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    parts.append(getattr(item, "text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content).strip()

    def generate(
        self, query: str, context_chunks: List[Tuple[DocumentChunk, float]]
    ) -> Tuple[str, float]:
        """Generate a response grounded in retrieved context chunks.
        
        Returns:
            Tuple of (answer_text, generation_latency_ms)
        """
        t0 = time.perf_counter()

        if not context_chunks:
            ans = "Based on the provided NASA technical reports, this information is not available."
            latency_ms = (time.perf_counter() - t0) * 1000
            return ans, latency_ms

        context_str = format_context_passages(context_chunks)
        user_content = USER_PROMPT_TEMPLATE.format(
            context_passages=context_str,
            query=query,
        )

        if self.llm is not None:
            try:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]
                response = self.llm.invoke(messages)
                answer_text = self._extract_text(response.content)
                latency_ms = (time.perf_counter() - t0) * 1000
                return answer_text, latency_ms
            except Exception as e:
                logger.warning("Gemini invocation failed (%s), using extractive fallback", e)

        # Extractive fallback generator for offline / fallback mode
        latency_ms = (time.perf_counter() - t0) * 1000
        answer_text = self._extractive_fallback(query, context_chunks)
        return answer_text, latency_ms

    def _extractive_fallback(
        self, query: str, context_chunks: List[Tuple[DocumentChunk, float]]
    ) -> str:
        """Produce a grounded extractive answer when LLM API is unavailable."""
        top_chunk, score = context_chunks[0]
        sentences = [s.strip() for s in top_chunk.text.split(".") if len(s.strip()) > 20]
        q_words = set(query.lower().split())

        best_sentences = []
        for s in sentences:
            s_words = set(s.lower().split())
            overlap = len(q_words.intersection(s_words))
            if overlap > 0:
                best_sentences.append((s, overlap))

        best_sentences.sort(key=lambda x: x[1], reverse=True)
        if best_sentences:
            extracted = " ".join([b[0] + "." for b in best_sentences[:2]])
            return f"{extracted} {top_chunk.citation_tag}"
        return f"{sentences[0]}. {top_chunk.citation_tag}"
