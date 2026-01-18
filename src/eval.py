from src.llm import llm

def evaluate(question, context, answer):
    prompt = f"""
You are an expert evaluator.

Score the following on a scale of 0–10.

1. Faithfulness: Is the answer fully supported by the context?
2. Relevance: Does the answer address the question?
3. Context Precision: Is the context useful and not noisy?

Return format:
Faithfulness: X
Relevance: X
Context Precision: X

Question:
{question}

Context:
{context}

Answer:
{answer}
"""
    return llm(prompt)
