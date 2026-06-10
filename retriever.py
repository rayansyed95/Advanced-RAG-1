import chromadb
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from query_transform import rewrite_query

# -------------------
# Config
# ------------------

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------
# Load ChromaDB
# -------------

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Load Documents
# ---------------

# Get total count to avoid the default 100 limit
count = collection.count()
data = collection.get(limit=count)
documents = data['documents']
ids = data['ids']

# Vectorizer Config.
# ================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),   # captures bigrams
    max_df=0.85,           # ignore very common terms
    min_df=1
)
tfidf_matrix = vectorizer.fit_transform(documents)

# -------------------
# Semantic Search
# -------------------

# def semantic_search(query, top_k=3):
#     query_embedding = EMBEDDING_MODEL.encode([query]).tolist()
#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=top_k
#     )
#     return results["documents"][0]


def semantic_search(query, top_k=3):
    query_embedding = EMBEDDING_MODEL.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    docs = results["documents"][0]
    distances = results["distances"][0]
    return list(zip(docs, distances))


# ----------------
# Keyword Seach
# ---------------
def keyword_search(query, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = (query_vec * tfidf_matrix.T).toarray()[0]

    # If no keywords match, return empty list instead of arbitrary documents
    if np.max(scores) == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]


# ------------
# Re-ranker
# ------------

def rerank_results(query, results, top_k=5):
    pairs = [(query, doc) for doc in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# --------------------
# Hybrid Search
# --------------------

# def hybrid_search(query, top_k=5):
#     rewritten_query = rewrite_query(query)
#     semantic_results = semantic_search(rewritten_query, top_k)
#     keyword_results = keyword_search(rewritten_query, top_k)
#     combined = list(dict.fromkeys(semantic_results + keyword_results))
#     return combined[:top_k], rewritten_query


def hybrid_search(query, top_k=5, k=60):
    rewritten_query = rewrite_query(query)

    semantic_results = semantic_search(rewritten_query, top_k=top_k*2)
    keyword_results = keyword_search(query, top_k=top_k*2)

    # ── Unpack (doc, distance) tuples from semantic_search ──────────────
    semantic_docs = [doc for doc, distance in semantic_results]
    semantic_distances = [distance for doc, distance in semantic_results]
    # ─────────────────────────────────────────────────────────────────────

    # Reciprocal Rank Fusion
    scores = {}
    for rank, doc in enumerate(semantic_docs):          # ← was: semantic_results
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    for rank, doc in enumerate(keyword_results):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    combined = [doc for doc, _ in ranked[:top_k]]

    return combined, rewritten_query
