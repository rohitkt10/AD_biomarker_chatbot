import os
from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import anthropic
from pdb import set_trace

# Paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"
os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
os.environ["HF_HOME"] = str(models_dir)

def load_model(model_name="Qwen/Qwen3-Embedding-8B"):
    return SentenceTransformer(model_name, device="cuda", trust_remote_code=True)

def load_index_assets():
    index = faiss.read_index(str(processed_dir / "faiss.index"))
    with open(processed_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(processed_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, chunks, metadata


def retrieve(query, model, index, chunks, metadata, k=5):
    """Top-k retrieval"""
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype("float32"), k)
    results = []
    for idx in I[0]:
        m = metadata[idx]
        results.append({
            "chunk": chunks[idx],
            "pmcid": m["pmcid"],
            "section_index": m["section_index"],
            "section_title": m["section_title"],
            "chunk_index_in_section": m["chunk_index_in_section"],
            "faiss_id": idx
        })
    return results


def answer_query(question, k=5, model_name="Qwen/Qwen3-Embedding-8B"):
    # Load assets
    emb_model = load_model(model_name=model_name)
    index, chunks, metadata = load_index_assets()

    # Retrieve chunks
    results = retrieve(question, emb_model, index, chunks, metadata, k=k)

    # Build context for Claude
    context_blocks = []
    for r in results:
        header = f"(PMCID={r['pmcid']}, Section={r['section_title']})"
        context_blocks.append(header + "\n" + r["chunk"])
    context = "\n\n---\n\n".join(context_blocks)
    context += "---\n"

    # Claude client
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    content = "Use the provided excerpts from \
        Alzheimer's research to answer the question that follows.\n\n" \
            + f"Context:\n{context}\n\nQuestion: {question}"
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
    return msg.content[0].text, results

if __name__ == "__main__":
    # example query 
    # this is a sanity check. PMC11350031 is an exact match for this question and the retrieved 
    # sources should be all or mostly from PMC11350031
    q = "What biomarkers are associated with gut microbiome changes in Alzheimer's disease?"
    answer, sources = answer_query(q, k=5)
    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES ===\n")
    for s in sources:
        print(s["pmcid"], s["section_title"])