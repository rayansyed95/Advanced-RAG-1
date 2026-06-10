from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from query_transform import rewrite_query

# -------------------
# Config
# -------------------

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "rag_collection"

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------
# Load ChromaDB
# -------------------

print("=" * 50)
print("Loading ChromaDB")
print("Chroma Path:", CHROMA_PATH)
print("=" * 50)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# -------------------
# Load Documents
# -------------------

count = collection.count()

print("Collection:", COLLECTION_NAME)
print("Collection Count:", count)

documents = []
ids = []

if count > 0:
    data = collection.get(limit=count)

    documents = data.get("documents", [])
    ids = data.get("ids", [])

    # Remove empty documents
    documents = [
        doc.strip()
        for doc in documents
        if doc and isinstance(doc, str) and doc.strip()
    ]

print("Documents Loaded:", len(documents))

if len(documents) > 0:
    print("First Document Preview:")
    print(documents[0][:300])

# -------------------
# TF-IDF Setup
# -------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.85,
    min_df=1
)

tfidf_matrix = None

if documents:
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
    except Exception as e:
        print("TF-IDF Error:", e)
else:
    print("WARNING: No documents found for TF-IDF indexing.")

print("=" * 50)

# -------------------
# Semantic Search
# -------------------


def semantic_search(query, top_k=3):

    if collection.count() == 0:
        return []

    query_embedding = EMBEDDING_MODEL.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    return list(zip(docs, distances))

# -------------------
# Keyword Search
# -------------------


def keyword_search(query, top_k=3):

    if tfidf_matrix is None:
        return []

    query_vec = vectorizer.transform([query])

    scores = (
        query_vec * tfidf_matrix.T
    ).toarray()[0]

    if len(scores) == 0:
        return []

    if np.max(scores) == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [documents[i] for i in top_indices]

# -------------------
# Re-ranker
# -------------------


def rerank_results(query, results, top_k=5):

    if not results:
        return []

    pairs = [
        (query, doc)
        for doc in results
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, _
        in ranked[:top_k]
    ]

# -------------------
# Hybrid Search
# -------------------

# def hybrid_search(query, top_k=5):
#     rewritten_query = rewrite_query(query)
#     semantic_results = semantic_search(rewritten_query, top_k)
#     keyword_results = keyword_search(rewritten_query, top_k)
#     combined = list(dict.fromkeys(semantic_results + keyword_results))
#     return combined[:top_k], rewritten_query


def hybrid_search(query, top_k=5, k=60):

    rewritten_query = rewrite_query(query)

    semantic_results = semantic_search(
        rewritten_query,
        top_k=top_k * 2
    )

    keyword_results = keyword_search(
        rewritten_query,
        top_k=top_k * 2
    )

    semantic_docs = [
        doc
        for doc, distance in semantic_results
    ]

    scores = {}

    # Reciprocal Rank Fusion
    for rank, doc in enumerate(semantic_docs):
        scores[doc] = (
            scores.get(doc, 0)
            + 1 / (k + rank)
        )

    for rank, doc in enumerate(keyword_results):
        scores[doc] = (
            scores.get(doc, 0)
            + 1 / (k + rank)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    combined = [
        doc
        for doc, _
        in ranked[:top_k]
    ]

    return combined, rewritten_query
