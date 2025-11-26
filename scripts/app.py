import os
import pickle
from pathlib import Path
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import anthropic
from query import load_model, load_index_assets, retrieve, answer_query

# Paths and environment setup
script_dir = Path(__file__).parent
project_root = script_dir.parent
processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"
os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
os.environ["HF_HOME"] = str(models_dir)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")



# Load embedding model, FAISS, chunks, metadata
@st.cache_resource(show_spinner=False)
def load_assets():
    model = load_model()
    index, chunks, metadata = load_index_assets()
    return model, index, chunks, metadata

# Streamlit Chat UI
def main():
    st.set_page_config(page_title="AD biomarkers RAG Chatbot", layout="wide")
    st.title("AD biomarkers RAG Chatbot")
    if ANTHROPIC_API_KEY is None:
        st.error("Missing ANTHROPIC_API_KEY environment variable.")
        return
    
    # load assets 
    model, index, chunks, metadata = load_assets()

    if "history" not in st.session_state:
        st.session_state.history = []

    # Render existing messages
    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.write(content)
    
    # Chat input
    user_msg = st.chat_input("Ask a question about biomarkers of Alzheimer's disease…")
    if not user_msg:
        return

    # Display user message
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.write(user_msg)
    
    
    with st.chat_message("assistant"):
        # Retrieval
        with st.spinner("Retrieving relevant chunks…"):
            results = retrieve(
                            user_msg,
                            model=model,
                            index=index,
                            chunks=chunks,
                            metadata=metadata,
                            k=5,
                            )
    
        # Generation
        with st.spinner("Generating answer…"):
            # Build context for Claude
            context_blocks = []
            for r in results:
                header = f"(PMCID={r['pmcid']}, Section={r['section_title']})"
                context_blocks.append(header + "\n" + r["chunk"])
            context = "\n\n---\n\n".join(context_blocks)
            context += "---\n"

            # set up client and 
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            content = "Use the provided excerpts from \
                Alzheimer's research to answer the question that follows.\n\n" \
                    + f"Context:\n{context}\n\nQuestion: {user_msg}"
            msg = client.messages.create(
                                    model="claude-sonnet-4-5",
                                    max_tokens=1024,
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": content,
                                        }
                                    ]
                                    )
            answer = msg.content[0].text
        
        st.write(answer)
        st.session_state.history.append(("assistant", answer))

        # Sources
        st.markdown("### Sources")
        pmcids = list(set([r["pmcid"] for r in results]))
        out_str = ""
        for id in pmcids:
            out_str + f"{id}, "
        out_str = out_str[:-2]
        st.markdown(out_str)

if __name__ == "__main__":
    main()
