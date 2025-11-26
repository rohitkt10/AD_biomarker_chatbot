import argparse
import time
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from Bio import Entrez


# PATHS 
script_dir = Path(__file__).parent
project_root = script_dir.parent
papers_dir = project_root / "data" / "papers"
papers_dir.mkdir(parents=True, exist_ok=True)

Entrez.email = "rohit.tripathy@jax.org"

def search_pubmed(query, n_pmids):
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=n_pmids,
        sort="relevance",
    )
    res = Entrez.read(handle)
    handle.close()
    return res["IdList"]


def get_pmc_id_from_pmid(pmid):
    h = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pmc")
    res = Entrez.read(h)
    h.close()
    if res and res[0].get("LinkSetDb"):
        return res[0]["LinkSetDb"][0]["Link"][0]["Id"]
    return None

def fetch_or_load_xml(pmc_id):
    xml_path = papers_dir / f"PMC{pmc_id}.xml"
    if xml_path.exists():
        with open(xml_path, "rb") as f:
            return f.read()
    handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
    xml_bytes = handle.read()
    handle.close()
    with open(xml_path, "wb") as f:
        f.write(xml_bytes)
    return xml_bytes

def parse_xml(raw_bytes):
    return ET.fromstring(raw_bytes)

def extract_title_abstract(root):
    title = root.find(".//article-title")
    abstract = root.find(".//abstract")

    out = []
    if title is not None:
        t = "".join(title.itertext()).strip()
        out.append("TITLE: " + t)

    if abstract is not None:
        a = "".join(abstract.itertext()).strip()
        out.append("ABSTRACT: " + a)

    return "\n".join(out) if out else ""

def extract_body(root):
    """
    Return formatted body text.
    If <body> is missing → return "".
    """
    body = root.find(".//body")
    if body is None:
        return ""

    sections = []

    # Only top-level <sec> under body
    for sec in body.findall("./sec"):
        sec_parts = []

        title_elem = sec.find("title")
        if title_elem is not None:
            sec_title = "".join(title_elem.itertext()).strip()
            if sec_title:
                sec_parts.append(sec_title.upper())

        for p in sec.findall(".//p"):
            text = "".join(p.itertext()).strip()
            if text:
                sec_parts.append(text)

        if sec_parts:
            sections.append("\n".join(sec_parts))

    return "\n\n".join(sections)

def extract_metadata(root, pmc_id):
    md = {"pmc_id": pmc_id}
    journal = root.find(".//journal-title")
    md["journal"] = journal.text if journal is not None else None
    pub_date = root.find(".//pub-date[@pub-type='epub']") or root.find(".//pub-date")
    if pub_date is not None:
        year = pub_date.find("year")
        month = pub_date.find("month")
        md["year"] = year.text if year is not None else None
        md["month"] = month.text if month is not None else None
    authors = []
    for contrib in root.findall(".//contrib[@contrib-type='author']"):
        sn = contrib.find(".//surname")
        gn = contrib.find(".//given-names")
        if sn is not None:
            authors.append(f"{gn.text if gn is not None else ''} {sn.text}".strip())
    md["authors"] = authors
    doi = root.find(".//article-id[@pub-id-type='doi']")
    md["doi"] = doi.text if doi is not None else None
    return md

def process_pmc(pmc_id):
    try:
        raw = fetch_or_load_xml(pmc_id)
        root = parse_xml(raw)

        title_abs = extract_title_abstract(root)
        body_text = extract_body(root)  # may be ""

        blocks = [b for b in [title_abs, body_text] if b.strip()]
        full_text = "\n\n".join(blocks)

        text_path = papers_dir / f"PMC{pmc_id}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        metadata = extract_metadata(root, pmc_id)
        meta_path = papers_dir / f"PMC{pmc_id}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Processed PMC{pmc_id}")
        return True

    except Exception as e:
        print(f"Failed PMC{pmc_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50,
                        help="Number of final papers to download (default=50)")
    args = parser.parse_args()
    N = args.n
    query_n = N * 3  # factor of 3
    query = "(Alzheimer's disease) AND (biomarkers) AND (2023:2025[dp])"
    print(f"Searching PubMed for {query_n} PMIDs ...")
    pmids = search_pubmed(query, query_n)
    print(f"Found {len(pmids)} PMIDs")
    count = 0
    for pmid in pmids:
        pmc_id = get_pmc_id_from_pmid(pmid)
        if pmc_id is None:
            continue
        ok = process_pmc(pmc_id)
        if ok:
            count += 1
        time.sleep(0.5)
        if count >= N:
            break
    print(f"Done. Extracted {count} PMC articles.")

if __name__ == "__main__":
    main()