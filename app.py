from src.rag import ask

while True:
    q = input("Ask Auto-Analyst: ")
    print(ask(q))
