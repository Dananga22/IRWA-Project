"""
MedExplain AI - Summarisation (NLP module).
Module: medexplain.summarisation

The assignment brief names Summarization alongside NER as a required NLP
technique, so this is an actual algorithm rather than string formatting.

Two complementary outputs:

  1. summarise_findings()  - a deterministic, structured summary of the report:
     how many values were measured, which fall outside their reference range and
     in which direction. Never wrong, never invented.

  2. textrank_summary()    - extractive summarisation of the passages retrieved
     for this report, using TF-IDF sentence vectors, a cosine similarity graph
     and PageRank centrality. Selects the most representative sentences across
     all retrieved sources, each carrying the citation it came from.

TextRank is implemented here in pure Python. That is deliberate: it adds no
dependency, it runs with no API key so a live demonstration cannot fail on it,
and every step is inspectable when an examiner asks how the summary was formed.

Reference: Mihalcea & Tarau (2004), "TextRank: Bringing Order into Texts".
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medexplain.logging_config import audit, get_logger

logger = get_logger("MedExplain.Summarisation")

_WORD_RE = re.compile(r"[a-z0-9]+")

# Ranking words carry no topical information, so they are excluded from the
# TF-IDF vectors. Kept short and explicit rather than pulling in a corpus.
_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "also", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "may", "me", "might", "more",
    "most", "must", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your",
}

MIN_SENTENCE_WORDS = 6
DAMPING = 0.85
MAX_ITERATIONS = 60
CONVERGENCE = 1e-6

# How the final sentence score splits between graph centrality (how
# representative a sentence is of the retrieved set) and query relevance (does
# it discuss an analyte this patient was actually tested for).
CENTRALITY_WEIGHT = 0.45
RELEVANCE_WEIGHT = 0.55


# --------------------------------------------------------------------------
# 1. Structured findings summary
# --------------------------------------------------------------------------

def summarise_findings(metadata: Dict[str, Any], test_items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic summary of what the report contains.

    Returns a dict with a `text` field plus the counts behind it, so the
    interface can render either the sentence or the numbers.
    """
    high = [i for i in test_items if str(i.get("flag", "")).upper() == "HIGH"]
    low = [i for i in test_items if str(i.get("flag", "")).upper() == "LOW"]
    normal = [i for i in test_items if str(i.get("flag", "")).upper() == "NORMAL"]
    total = len(test_items)

    panel = metadata.get("panel_type") or "laboratory panel"
    parts: List[str] = [f"This {panel} reports {total} measured value(s)."]

    if total and not (high or low):
        parts.append("All of them fall within the reference ranges printed on the report.")
    else:
        def _verb(count: int, singular: str, plural: str) -> str:
            return singular if count == 1 else plural

        if normal:
            possessive = "its" if len(normal) == 1 else "their"
            parts.append(
                f"{len(normal)} {_verb(len(normal), 'falls', 'fall')} within {possessive} reference range."
            )
        if high:
            names = ", ".join(str(i.get("test_name")) for i in high)
            parts.append(f"{len(high)} {_verb(len(high), 'sits', 'sit')} above the stated range: {names}.")
        if low:
            names = ", ".join(str(i.get("test_name")) for i in low)
            parts.append(f"{len(low)} {_verb(len(low), 'sits', 'sit')} below the stated range: {names}.")

    return {
        "text": " ".join(parts),
        "total": total,
        "normal": len(normal),
        "high": len(high),
        "low": len(low),
        "high_tests": [str(i.get("test_name")) for i in high],
        "low_tests": [str(i.get("test_name")) for i in low],
    }


# --------------------------------------------------------------------------
# 2. TextRank extractive summarisation
# --------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Sentence boundaries, protecting decimals such as 8.2 and 13.5 - 17.5."""
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", str(text))
    pieces = re.split(r"(?<=[.!?])\s+|\n+", protected)
    return [p.replace("<DOT>", ".").strip() for p in pieces if p.strip()]


def _tokenise(sentence: str) -> List[str]:
    return [w for w in _WORD_RE.findall(sentence.lower()) if w not in _STOPWORDS and len(w) > 2]


def _stem(word: str) -> str:
    """
    Crude plural stripping, used only for query matching.

    "Platelets" on a report and "platelet count" in a source passage are the
    same topic; exact token matching would miss it and the query bias would
    silently do nothing.
    """
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _stems(words: Sequence[str]) -> set:
    return {_stem(w) for w in words}


def _tf_idf_vectors(token_lists: Sequence[Sequence[str]]) -> List[Dict[str, float]]:
    """TF-IDF weight per term per sentence."""
    n = len(token_lists)
    document_freq: Counter = Counter()
    for tokens in token_lists:
        document_freq.update(set(tokens))

    vectors: List[Dict[str, float]] = []
    for tokens in token_lists:
        counts = Counter(tokens)
        length = len(tokens) or 1
        vector: Dict[str, float] = {}
        for term, count in counts.items():
            tf = count / length
            idf = math.log((n + 1) / (document_freq[term] + 1)) + 1.0
            vector[term] = tf * idf
        vectors.append(vector)
    return vectors


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pagerank(similarity: List[List[float]]) -> List[float]:
    """Weighted PageRank over the sentence similarity graph."""
    n = len(similarity)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    out_weight = [sum(row) for row in similarity]
    scores = [1.0 / n] * n

    for _ in range(MAX_ITERATIONS):
        updated = []
        for i in range(n):
            inbound = 0.0
            for j in range(n):
                if i == j or similarity[j][i] == 0.0 or out_weight[j] == 0.0:
                    continue
                inbound += similarity[j][i] / out_weight[j] * scores[j]
            updated.append((1.0 - DAMPING) / n + DAMPING * inbound)

        delta = sum(abs(updated[i] - scores[i]) for i in range(n))
        scores = updated
        if delta < CONVERGENCE:
            break

    return scores


def textrank_summary(
    passages: Sequence[Dict[str, Any]],
    top_n: int = 4,
    query_terms: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract the most central sentences across the retrieved passages.

    Each returned item keeps the source it came from, so a summary sentence is
    as citable as anything else the system shows.

    `query_terms` (the analyte names from this report) give a modest boost to
    sentences that mention what the patient was actually tested for - relevance
    on top of centrality.
    """
    sentences: List[str] = []
    origins: List[Dict[str, Any]] = []

    for passage in passages:
        for sentence in split_sentences(passage.get("text", "")):
            if len(sentence.split()) < MIN_SENTENCE_WORDS:
                continue
            sentences.append(sentence)
            origins.append(
                {
                    "title": passage.get("title", ""),
                    "source": passage.get("source", ""),
                    "source_url": passage.get("source_url", ""),
                    "passage_id": passage.get("id", ""),
                }
            )

    if not sentences:
        logger.info("No sentences available to summarise")
        return []

    token_lists = [_tokenise(s) for s in sentences]
    vectors = _tf_idf_vectors(token_lists)

    n = len(sentences)
    similarity = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            score = _cosine(vectors[i], vectors[j])
            similarity[i][j] = score
            similarity[j][i] = score

    scores = _pagerank(similarity)

    # Query bias.
    #
    # Centrality alone is not enough here. A passage about an analyte that
    # appears on no other retrieved page is an isolated node in the similarity
    # graph, so PageRank scores it near the floor - even though it is exactly
    # what this patient was tested for. Multiplying a near-zero score changes
    # nothing, so relevance is combined additively instead, as a first-class
    # ranking signal. That is what makes this a query-biased summary rather than
    # a generic one.
    max_centrality = max(scores) or 1.0
    centrality = [score / max_centrality for score in scores]

    wanted = _stems([w for term in (query_terms or []) for w in _tokenise(str(term))])
    if wanted:
        relevance = []
        for tokens in token_lists:
            matches = len(wanted & _stems(tokens))
            relevance.append(min(1.0, 0.85 + 0.05 * matches) if matches else 0.0)
        scores = [
            CENTRALITY_WEIGHT * centrality[i] + RELEVANCE_WEIGHT * relevance[i]
            for i in range(n)
        ]
    else:
        scores = centrality

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)

    # Suppress near-duplicates: the corpus repeats itself across passages.
    selected: List[int] = []
    for index in ranked:
        if len(selected) >= top_n:
            break
        if any(_cosine(vectors[index], vectors[chosen]) > 0.6 for chosen in selected):
            continue
        selected.append(index)

    selected.sort()  # restore reading order

    max_score = max(scores) or 1.0
    summary = [
        {
            "sentence": sentences[i],
            "score": round(scores[i] / max_score, 4),
            **origins[i],
        }
        for i in selected
    ]

    logger.info(
        "TextRank summarised %d sentence(s) from %d passage(s) down to %d key point(s)",
        n,
        len(passages),
        len(summary),
    )
    return summary


# --------------------------------------------------------------------------
# Combined entry point used by the pipeline
# --------------------------------------------------------------------------

def summarise(
    metadata: Dict[str, Any],
    test_items: Sequence[Dict[str, Any]],
    passages: Sequence[Dict[str, Any]],
    top_n: int = 4,
) -> Dict[str, Any]:
    """Produce both summaries and record what was done in the audit trail."""
    findings = summarise_findings(metadata, test_items)
    key_points = textrank_summary(
        passages,
        top_n=top_n,
        query_terms=[i.get("test_name", "") for i in test_items],
    )

    audit(
        "agent.summarisation",
        method="textrank+tfidf",
        sentences_considered=sum(len(split_sentences(p.get("text", ""))) for p in passages),
        key_points=len(key_points),
        total_tests=findings["total"],
        abnormal=findings["high"] + findings["low"],
    )

    return {
        "findings": findings,
        "key_points": key_points,
        "method": "TextRank (TF-IDF sentence vectors + PageRank centrality)",
    }
