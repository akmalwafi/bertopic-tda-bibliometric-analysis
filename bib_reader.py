# bib_reader.py
import bibtexparser


def load_bib(path: str):
    """
    Load one BibTeX file and return BibDatabase object
    """
    with open(path, encoding="utf-8") as f:
        bib_db = bibtexparser.load(f)
    return bib_db


def load_multiple_bibs(paths: list[str]):
    """
    Load multiple BibTeX files and combine all entries into one list
    """
    all_entries = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            bib_db = bibtexparser.load(f)
            all_entries.extend(bib_db.entries)

    return all_entries


def bib_to_documents(entries, include_keywords: bool = False):
    """
    Convert entries → list of raw documents
    (works for BOTH single and multiple files)
    """
    docs = []

    for entry in entries:
        title = entry.get("title", "")
        abstract = entry.get("abstract", "")
        keywords = entry.get("keywords", "") if include_keywords else ""

        raw_text = " ".join([title, abstract, keywords]).strip()
        docs.append(raw_text)

    return docs


def bib_to_entries(bib_db):
    """
    Return raw entries (for single file)
    """
    return bib_db.entries