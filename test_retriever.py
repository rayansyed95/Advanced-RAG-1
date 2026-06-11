from retriever import RAGRetriever

retriever = RAGRetriever()

while True:
    query = input("Ask a question: ")
    results, rewritten_query = retriever.hybrid_search(query)
    print(f"\nRewritten Query: {rewritten_query}")
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result[:200]}...\n")
    if query.lower() == "exit":
        break