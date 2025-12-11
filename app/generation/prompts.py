
"""Prompt templates for medical text simplification."""

SYSTEM_PROMPT = """You are a helpful medical communication assistant. Your job is to help patients understand their medical information by explaining it in simple, clear language.

Guidelines:
- Use simple words that an 8th grader can understand
- Avoid medical jargon - if you must use a medical term, explain it
- Use short sentences (under 20 words when possible)
- Be warm and reassuring, but accurate
- Use "you" and "your" to speak directly to the patient
- Break complex information into small, digestible pieces
- Use everyday comparisons when helpful"""

SIMPLIFY_PROMPT = """Please simplify the following medical text so that a patient with an 8th-grade reading level can easily understand it. 

Keep all the important medical information but explain it in plain, simple language. If there are medical terms, briefly explain what they mean.

Medical text to simplify:
---
{text}
---

Simplified version:"""

ANSWER_PROMPT = """A patient is asking a question about their medical care. Use the following information from their medical documents to answer their question in simple, easy-to-understand language.

Patient's question: {question}

Relevant information from their medical records:
---
{context}
---

Please answer the patient's question using the information above. Remember to:
- Use simple language (8th-grade reading level)
- Be clear and direct
- If the information doesn't fully answer their question, say so
- Include any important warnings or things they should watch for

Answer:"""

READABILITY_CHECK_PROMPT = """Review this simplified medical text. Is it easy enough for someone with an 8th-grade education to understand?

Text:
---
{text}
---

If the text is too complex, rewrite it to be simpler. If it's already simple enough, return it unchanged.

Final simplified text:"""