import streamlit as st
from openai import OpenAI
import uuid
import os
import chromadb
from retriever import RAGRetriever, EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_PATH
from build_kb import extract_text_from_pdf, chunk_text
from dotenv import load_dotenv

load_dotenv()

# CONFIG
client = OpenAI()

st.set_page_config(page_title="Advanced RAG System",
                   layout="wide", page_icon="🤖")

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    }
    .question-card {
        background-color: black;
        color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4facfe;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .answer-card {
        background-color: black;
        color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00f2fe;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Advanced RAG System")
st.markdown("### Hybrid Search + Query Transformation + Semantic Reranking")

# Create history session if already not there.
if "history" not in st.session_state:
    st.session_state.history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "knowledge_source" not in st.session_state:
    st.session_state.knowledge_source = "Default Database"

if "retriever" not in st.session_state:
    st.session_state.retriever = RAGRetriever(COLLECTION_NAME)

MAX_QUESTIONS = 10

# Sidebar for stats and info

with st.sidebar:

    st.header("📂 Knowledge Source")

    source_choice = st.radio(
        "Select Source:",
        options=["Default Database", "Upload Document"],
        index=0 if st.session_state.knowledge_source == "Default Database" else 1
    )

    if source_choice != st.session_state.knowledge_source:
        st.session_state.knowledge_source = source_choice
        if source_choice == "Default Database":
            st.session_state.retriever = RAGRetriever(COLLECTION_NAME)
            st.success("Switched to Default Database!")
            st.rerun()
        else:
            user_collection_name = f"user_col_{st.session_state.session_id}"
            st.session_state.retriever = RAGRetriever(user_collection_name)
            st.rerun()

    if st.session_state.knowledge_source == "Upload Document":
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file:",
            type=["pdf", "txt"],
            key="document_uploader"
        )

        if uploaded_file is not None:
            st.info(f"File: {uploaded_file.name}")
            if st.button("Ingest Document ⚡", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        file_extension = os.path.splitext(
                            uploaded_file.name)[1].lower()
                        if file_extension == ".pdf":
                            text = extract_text_from_pdf(uploaded_file)
                        elif file_extension == ".txt":
                            text = uploaded_file.read().decode("utf-8")
                        else:
                            st.error("Unsupported file format.")
                            text = ""

                        if text:
                            chunks = chunk_text(text)
                            if not chunks:
                                st.warning(
                                    "No text could be extracted or chunked from the document.")
                            else:
                                embeddings = EMBEDDING_MODEL.encode(
                                    chunks).tolist()
                                user_collection_name = f"user_col_{st.session_state.session_id}"
                                client = chromadb.PersistentClient(
                                    path=CHROMA_PATH)
                                try:
                                    client.delete_collection(
                                        name=user_collection_name)
                                except Exception:
                                    pass
                                collection = client.get_or_create_collection(
                                    name=user_collection_name)
                                for i, chunk in enumerate(chunks):
                                    collection.add(
                                        ids=[str(i)],
                                        documents=[chunk],
                                        embeddings=[embeddings[i]],
                                        metadatas=[
                                            {"chunk_id": i, "source": uploaded_file.name}]
                                    )
                                st.session_state.retriever = RAGRetriever(
                                    user_collection_name)
                                st.success(
                                    "Document ingested successfully! You can now query it.")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error processing document: {e}")

    st.markdown("---")

    st.header("📊 Session Statistics")
    questions_asked = st.session_state.question_count
    questions_left = MAX_QUESTIONS - questions_asked

    st.write(f"**Questions Asked:** {questions_asked} / {MAX_QUESTIONS}")
    st.write(f"**Questions Left:** {questions_left}")

    # Progress Bar
    progress = min(questions_asked / MAX_QUESTIONS, 1.0)
    st.progress(progress)

    if questions_left <= 2 and questions_left > 0:
        st.warning(f"Only {questions_left} questions left!")
    elif questions_left == 0:
        st.error("Session limit reached.")

    if st.button("Reset Session"):
        st.session_state.history = []
        st.session_state.question_count = 0
        if st.session_state.knowledge_source == "Upload Document":
            user_collection_name = f"user_col_{st.session_state.session_id}"
            try:
                client = chromadb.PersistentClient(path=CHROMA_PATH)
                client.delete_collection(name=user_collection_name)
            except Exception:
                pass
            st.session_state.retriever = RAGRetriever(user_collection_name)
        st.rerun()

    st.markdown("---")


# Answering


def generate_response(query, context_chunks, history_context=""):
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are a helpful assistant.
    Use the context provided to answer the question.
    If the context does not match with the query, say "Sorry context not found" or suggest some similar follow up questions to get the context right.
    
    Previous Conversation History (Last 3 Interactions):
    {history_context}

    Current Context:
    {context}
    
    Question: {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content

# UI for Input


if st.session_state.question_count < MAX_QUESTIONS:
    has_docs = len(st.session_state.retriever.documents) > 0
    if not has_docs:
        st.info("👈 Please upload and ingest a document in the sidebar to start asking questions, or switch to the Default Database.")
    else:
        with st.container():
            with st.form(key="query_form", clear_on_submit=True):
                st.markdown("#### ASK YOUR QUERY")
                input_query = st.text_input(
                    "Enter your question about the documents:", placeholder="e.g., What is the main topic?")
                submit_button = st.form_submit_button("Send Query 📤")

        if submit_button and input_query:
            with st.spinner("🔍 Analysing documents and generating answer..."):
                # Prepare history context (last 3 interactions: questions + answers)
                last_3_items = st.session_state.history[-3:]
                history_context = ""
                for item in last_3_items:
                    # Here is one problem
                    history_context += f"Question: {item['query']}\nAnswer: {item['answer']}\n\n"

                # Retrieval & Transform
                chunks, transformed_query = st.session_state.retriever.hybrid_search(
                    input_query)
                reranked_chunks = st.session_state.retriever.rerank_results(
                    input_query, chunks)

                # Generation
                answer = generate_response(
                    input_query, reranked_chunks, history_context)

                # Save to history
                st.session_state.history.append({
                    "query": input_query,
                    "answer": answer,
                    "transformed": transformed_query,
                    "chunks": chunks
                })
                st.session_state.question_count += 1
                st.rerun()  # Rerun to refresh UI and counter
else:
    st.error(
        "🛑 **Limit Reached:** You have asked 10 questions. Please reset the session to continue.")

# Display Chat History (and latest answer)
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 💬 Conversation History")

    # Show history in reverse order so latest is on top
    for i, item in enumerate(reversed(st.session_state.history)):
        is_latest = (i == 0)

        with st.container():
            st.markdown(
                f'<div class="question-card"><strong>Q: {item["query"]}</strong></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-card"><strong>A:</strong><br>{item["answer"]}</div>', unsafe_allow_html=True)

            # Expander for debug info
            with st.expander(f"🔍 Technical Details {'(Latest)' if is_latest else ''}"):
                st.write(f"**Transformed Query:** {item['transformed']}")
                st.write("**Retrieved Chunks:**")
                for j, chunk in enumerate(item['chunks']):
                    st.info(f"Chunk {j+1}: {chunk[:500]}...")

        if is_latest:
            st.markdown("### Previous Messages")

else:
    st.info("Start a conversation by asking a question above.")
