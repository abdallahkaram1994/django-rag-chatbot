from src.rag import answer

for q in ("How do I create a custom user model?", "What is the capital of France?"):
    r = answer(q)
    print("\nQ:", q)
    print("A:", r["answer"])
    for s in r["sources"]:
        print(f"   [{s['n']}] {s['distance']}  {s['url']}  |  {s['section']}")
