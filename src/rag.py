from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.llm import llm
from src.eval import evaluate

embed = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embed)
retriever = db.as_retriever(search_kwargs={"k": 6})


def classify_intent(question: str) -> str:
    prompt = f"""
Classify the user's intent.

Return only ONE word: "research" or "chat"

Question: {question}
Answer:
"""
    return llm(prompt).strip().lower()


def ask(question: str):
    intent = classify_intent(question)

    # -------- CHAT MODE --------
    if "chat" in intent:
        return llm(question)

    # -------- RESEARCH MODE --------
    docs = retriever.invoke(question)

    filtered = [
        d for d in docs
        if "references" not in d.page_content.lower()
        and "bibliography" not in d.page_content.lower()
    ]

    context = "\n\n".join(
        f"(Page {d.metadata.get('page','?')}) {d.page_content}"
        for d in filtered
    )

    prompt = f"""
You are Auto-Analyst, an AI research assistant.

Use ONLY the main body of the paper to answer.

Context:
{context}

Question:
{question}

Answer clearly and then list the sources with page numbers.
"""

    answer = llm(prompt)
    scores = evaluate(question, context, answer)

    return answer

