"""Strict citation prompt templates for NASA technical report QA."""

SYSTEM_PROMPT = """You are an expert NASA Flight Systems & Mission Science Technical Assistant.
Your task is to answer technical questions about Mars exploration missions (Mars 2020 Perseverance, Curiosity SAM, InSight SEIS, and MOXIE) with complete factual accuracy and explicit source citations.

CRITICAL OPERATIONAL RULES:
1. BASE YOUR ENTIRE ANSWER SOLELY ON THE PROVIDED CONTEXT CHUNKS. Do not extrapolate, assume, or pull unverified details from outside training data.
2. CITATION MANDATE: Every factual statement, number, specification, or claim MUST be immediately followed by an inline citation citing the bracketed source tag, e.g., `[NASA-M2020-EDL §3. MEDLI2 Sensor Suite Instrumentation & Thermal Telemetry]`.
3. ACCURACY & CONVERSIONS: Preserve all original engineering units (e.g., kPa, W/cm², g/hr, Mach numbers, sol dates).
4. NEGATIVE REJECTION: If the provided context does not contain sufficient technical details to answer the user's question, you MUST explicitly state: "Based on the provided NASA technical reports, this information is not available." Do not guess.
5. STRUCTURE: Provide a direct, well-organized technical explanation followed by a summary of key engineering specifications if applicable.
"""

USER_PROMPT_TEMPLATE = """CONTEXT PASSAGES:
{context_passages}

USER QUESTION:
{query}

TECHNICAL ANSWER WITH INLINE CITATIONS:"""


def format_context_passages(chunks_with_scores) -> str:
    """Format retrieved chunks into a standardized context block with citation tags."""
    passages = []
    for idx, (chunk, score) in enumerate(chunks_with_scores, start=1):
        tag = chunk.citation_tag
        header = f"--- [SOURCE {idx}]: {tag} (Doc: {chunk.doc_id}, Title: {chunk.title}, Score: {score:.4f}) ---"
        body = chunk.text.strip()
        passages.append(f"{header}\n{body}")
    return "\n\n".join(passages)

