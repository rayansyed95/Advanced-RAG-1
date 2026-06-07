from query_transform import rewrite_query

while True:
    query = input("User Query: ")
    rewritten_query = rewrite_query(query)
    print("Rewritten Query:", rewritten_query)
    print("-" * 50)
    if query.lower() == "exit":
        break