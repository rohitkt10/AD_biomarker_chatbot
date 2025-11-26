import argparse
import pickle
from pathlib import Path
from pdb import set_trace
import faiss
import numpy as np, os
import json
import torch 
from torch import nn
from sentence_transformers import SentenceTransformer

# Setup paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
papers_dir = project_root / "data" / "papers"
models_dir = project_root / "models"
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# set up HF env vars 
os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
os.environ["HF_HOME"] = str(models_dir)

def chunk_section(words, chunk_size=500, overlap=50):
    """
    Chunk a single section (already split into words).
    Returns list of text chunks.
    """
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks

def chunk_paper_text(text, chunk_size=500, overlap=50):
    """
    Split by section (double newline).
    Then chunk each section independently.
    Returns:
      - chunks: list[str]
      - section_titles: list[str]
      - section_ids: list[int]
      - chunk_ids_in_section: list[int]
    """
    sections = text.split("\n\n")
    all_chunks = []
    all_section_titles = []
    all_section_ids = []
    all_chunk_ids = []

    for sec_idx, sec in enumerate(sections):
        lines = sec.split("\n")
        lines = [l.strip() for l in lines if l.strip()]

        if not lines:
            continue

        # First line of a body section is a TITLE (UPPERCASE),
        # but for Title+Abstract block it's not uppercase.
        # Simpler rule: use the *first non-empty line* as title.
        sec_title = lines[0]
        words = sec.split()
        sec_chunks = chunk_section(words, chunk_size, overlap)
        for chunk_idx, ch in enumerate(sec_chunks):
            all_chunks.append(ch)
            all_section_titles.append(sec_title)
            all_section_ids.append(sec_idx)
            all_chunk_ids.append(chunk_idx)
    return all_chunks, all_section_titles, all_section_ids, all_chunk_ids

def load_paper_metadata(paper_stem):
    """
    Look for the PMCxxxx_metadata.json file and extract pmid + pmcid.
    Returns (pmcid, pmid) or (None, None) if missing.
    """
    meta_path = papers_dir / f"{paper_stem}_metadata.json"
    if not meta_path.exists():
        return None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        md = json.load(f)
    pmcid = md.get("pmc_id", None)
    pmid = md.get("pmid", None)  
    return pmcid, pmid

def load_and_chunk_papers(chunk_size=500, overlap=50):
    all_chunks = []
    metadata = []
    for txt_file in papers_dir.glob("PMC*.txt"):
        print(f"Processing {txt_file.name}...")
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        paper_stem = txt_file.stem  
        pmcid, pmid = load_paper_metadata(paper_stem)

        # chunking
        chunks, section_titles, section_ids, chunk_ids = chunk_paper_text(
            text, chunk_size=chunk_size, overlap=overlap
        )

        for i, ch in enumerate(chunks):
            md = {
                "pmcid": paper_stem,
                "section_index": section_ids[i],
                "section_title": section_titles[i],
                "chunk_index_in_section": chunk_ids[i],
            }
            all_chunks.append(ch)
            metadata.append(md)

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks, metadata

def build_faiss_index(embeddings, out_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(out_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-8B")
    args = parser.parse_args()

    print("Loading and chunking papers...")
    chunks, metadata = load_and_chunk_papers(
        chunk_size=args.chunk_size, overlap=args.overlap
    )

    print("Loading embedding model...")
    model = SentenceTransformer(args.model, device="cuda")

    print("Embedding chunks...")
    embeddings = model.encode(
                            chunks,
                            batch_size=8,
                            show_progress_bar=True,
                            )

    print("Building FAISS index...")
    build_faiss_index(
        embeddings,
        processed_dir / "faiss.index"
    )

    print("Saving chunks + metadata...")
    with open(processed_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    with open(processed_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    main()