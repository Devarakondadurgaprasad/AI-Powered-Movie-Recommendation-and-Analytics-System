from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - fallback for minimal local runtimes
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover - optional dependency fallback
    fuzz = None
    process = None


class MovieRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        if TfidfVectorizer is not None and cosine_similarity is not None:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combinedFeatures"])
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        else:
            self.vectorizer = SimpleTfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combinedFeatures"])
            self.similarity_matrix = simple_cosine_similarity(self.tfidf_matrix)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def search_titles(self, query: str, limit: int = 10) -> list[str]:
        titles = self.df["title"].tolist()
        if not query:
            return titles[:limit]
        query_lower = query.lower()
        partial = [title for title in titles if query_lower in title.lower()]
        if len(partial) >= limit or process is None:
            return partial[:limit]
        fuzzy = process.extract(query, titles, scorer=fuzz.WRatio, limit=limit)
        merged = partial + [match[0] for match in fuzzy if match[0] not in partial and match[1] >= 55]
        return merged[:limit]

    def recommend(
        self,
        title: str,
        top_n: int = 10,
        genre: str | None = None,
        year_range: tuple[int, int] | None = None,
        min_rating: float = 0.0,
        title_type: str | None = None,
    ) -> pd.DataFrame:
        if title not in set(self.df["title"]):
            matches = self.search_titles(title, 1)
            if not matches:
                return pd.DataFrame()
            title = matches[0]

        selected_index = self.df.index[self.df["title"].eq(title)][0]
        scores = list(enumerate(self.similarity_matrix[selected_index]))
        candidates = (
            pd.DataFrame(scores, columns=["index", "similarityScore"])
            .query("index != @selected_index")
            .merge(self.df, left_on="index", right_index=True)
        )

        if genre and genre != "All":
            candidates = candidates[candidates["genres"].str.contains(genre, case=False, regex=False)]
        if year_range:
            candidates = candidates[candidates["year"].between(year_range[0], year_range[1])]
        if min_rating:
            candidates = candidates[candidates["averageRating"].ge(min_rating)]
        if title_type and title_type != "All":
            candidates = candidates[candidates["titleType"].eq(title_type)]

        columns = ["title", "similarityScore", "averageRating", "numVotes", "genres", "year", "titleType"]
        return (
            candidates.sort_values(["similarityScore", "averageRating", "numVotes"], ascending=False)
            .head(top_n)[columns]
            .reset_index(drop=True)
        )


class SimpleMatrix(list):
    @property
    def shape(self) -> tuple[int, int]:
        if not self:
            return (0, 0)
        return (len(self), len(self[0]))


class SimpleTfidfVectorizer:
    def __init__(self):
        self.vocabulary_: dict[str, int] = {}

    def fit_transform(self, documents: pd.Series) -> SimpleMatrix:
        tokenized = [self._tokens(text) for text in documents]
        vocabulary = sorted({token for tokens in tokenized for token in tokens})
        self.vocabulary_ = {token: index for index, token in enumerate(vocabulary)}
        doc_count = len(tokenized)
        doc_freq = Counter(token for tokens in tokenized for token in set(tokens))
        matrix = SimpleMatrix()
        for tokens in tokenized:
            counts = Counter(tokens)
            row = [0.0] * len(vocabulary)
            for token, count in counts.items():
                index = self.vocabulary_[token]
                tf = count / max(len(tokens), 1)
                idf = math.log((1 + doc_count) / (1 + doc_freq[token])) + 1
                row[index] = tf * idf
            matrix.append(row)
        return matrix

    @staticmethod
    def _tokens(text: object) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", str(text).lower())


def simple_cosine_similarity(matrix: SimpleMatrix) -> SimpleMatrix:
    similarities = SimpleMatrix()
    norms = [math.sqrt(sum(value * value for value in row)) or 1.0 for row in matrix]
    for i, left in enumerate(matrix):
        row = []
        for j, right in enumerate(matrix):
            dot = sum(a * b for a, b in zip(left, right))
            row.append(dot / (norms[i] * norms[j]))
        similarities.append(row)
    return similarities
