from retriever import hybrid_search


while True:
    query = input("Ask a question: ")
    results = hybrid_search(query)
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result[:200]}...\n")
    if query.lower() == "exit":
        break