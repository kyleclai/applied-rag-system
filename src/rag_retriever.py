import os
import re
from typing import List, Tuple, Dict

# Borrowed from DocuBot (Module 4 Tinker) with minor adaptations
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "where", "when", "how", "what",
    "which", "who", "why", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "no", "if", "it", "its", "this", "that", "with",
    "from", "by", "as", "i", "me", "my", "we", "our", "you", "your",
}


class MusicKnowledgeRetriever:
    """TF-IDF-style retriever over the music_kb/ knowledge base."""

    def __init__(self, kb_folder: str = "music_kb"):
        self.kb_folder = kb_folder
        self.documents: List[Tuple[str, str]] = self.load_documents()
        self.index: Dict[str, List[str]] = self.build_index(self.documents)

    def load_documents(self) -> List[Tuple[str, str]]:
        """Load all .md and .txt files from kb_folder as (filename, text) pairs."""
        docs = []
        if not os.path.isdir(self.kb_folder):
            return docs
        for fname in sorted(os.listdir(self.kb_folder)):
            if fname.endswith(".md") or fname.endswith(".txt"):
                path = os.path.join(self.kb_folder, fname)
                with open(path, "r", encoding="utf-8") as f:
                    docs.append((fname, f.read()))
        return docs

    def build_index(self, documents: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Build inverted index: word -> list of filenames containing that word."""
        index: Dict[str, List[str]] = {}
        for fname, text in documents:
            for word in re.findall(r"[a-z]+", text.lower()):
                if word not in STOPWORDS:
                    index.setdefault(word, [])
                    if fname not in index[word]:
                        index[word].append(fname)
        return index

    def score_document(self, query: str, text: str) -> int:
        """Count meaningful (non-stopword) query words found in text."""
        words = set(re.findall(r"[a-z]+", query.lower())) - STOPWORDS
        return sum(1 for w in words if w in text.lower())

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[str, str]]:
        """Return top_k most relevant (source, chunk) pairs for the query."""
        query_words = set(re.findall(r"[a-z]+", query.lower())) - STOPWORDS
        if not query_words:
            return []

        # Score documents via inverted index
        doc_scores: Dict[str, int] = {}
        for word in query_words:
            for fname in self.index.get(word, []):
                doc_scores[fname] = doc_scores.get(fname, 0) + 1

        if not doc_scores:
            return []

        # Examine top 3 scoring documents, split into chunks, re-score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        doc_map = {fname: text for fname, text in self.documents}

        chunks: List[Tuple[int, str, str]] = []  # (score, source, text)
        for fname, _ in ranked_docs[:3]:
            text = doc_map.get(fname, "")
            for chunk in text.split("\n\n"):
                chunk = chunk.strip()
                if len(chunk) < 30:
                    continue
                score = self.score_document(query, chunk)
                if score > 0:
                    chunks.append((score, fname, chunk))

        chunks.sort(key=lambda x: x[0], reverse=True)
        return [(source, text) for _, source, text in chunks[:top_k]]
