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
# RAGRetriever Class
# -------------------

class RAGRetriever:
    def __init__(self, collection_name=COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        self.documents = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1
        )
        self.tfidf_matrix = None
        self.load_documents()

    def load_documents(self):
        count = self.collection.count()
        print(f"Loading documents for collection: {self.collection_name} (Count: {count})")
        if count > 0:
            data = self.collection.get(limit=count)
            self.documents = data.get("documents", [])
            # Remove empty documents
            self.documents = [
                doc.strip()
                for doc in self.documents
                if doc and isinstance(doc, str) and doc.strip()
            ]
        
        print("Documents Loaded:", len(self.documents))
        if self.documents:
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
                print("TF-IDF Matrix Shape:", self.tfidf_matrix.shape)
            except Exception as e:
                print("TF-IDF Error:", e)
        else:
            self.tfidf_matrix = None
            print("WARNING: No documents found for TF-IDF indexing.")

    def semantic_search(self, query, top_k=3):
        if self.collection.count() == 0:
            return []

        query_embedding = EMBEDDING_MODEL.encode(
            [query]
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count())
        )

        docs = results["documents"][0]
        distances = results["distances"][0]

        return list(zip(docs, distances))

    def keyword_search(self, query, top_k=3):
        if self.tfidf_matrix is None or not self.documents:
            return []

        query_vec = self.vectorizer.transform([query])

        scores = (
            query_vec * self.tfidf_matrix.T
        ).toarray()[0]

        if len(scores) == 0:
            return []

        if np.max(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [self.documents[i] for i in top_indices]

    def rerank_results(self, query, results, top_k=5):
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

    def hybrid_search(self, query, top_k=5, k=60):
        rewritten_query = rewrite_query(query)

        semantic_results = self.semantic_search(
            rewritten_query,
            top_k=top_k * 2
        )

        keyword_results = self.keyword_search(
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


# --------------------------------------------------
# Backward Compatibility - Module Level Interface
# --------------------------------------------------

_default_retriever = RAGRetriever(COLLECTION_NAME)

def semantic_search(query, top_k=3):
    return _default_retriever.semantic_search(query, top_k)

def keyword_search(query, top_k=3):
    return _default_retriever.keyword_search(query, top_k)

def rerank_results(query, results, top_k=5):
    return _default_retriever.rerank_results(query, results, top_k)

def hybrid_search(query, top_k=5, k=60):
    return _default_retriever.hybrid_search(query, top_k, k)

