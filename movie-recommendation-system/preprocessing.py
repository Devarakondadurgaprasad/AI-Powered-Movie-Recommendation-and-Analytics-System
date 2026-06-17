from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler
except Exception:  # pragma: no cover - fallback for minimal local runtimes
    MinMaxScaler = None


REQUIRED_COLUMNS = ["title", "titleType", "genres", "averageRating", "numVotes", "year"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load an IMDb-style CSV and validate the required schema."""
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return preprocess_data(df)


def clean_genres(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    cleaned = re.sub(r"\s+", " ", str(value).replace("|", ",")).strip()
    cleaned = cleaned.strip(",")
    return cleaned if cleaned and cleaned != "\\N" else "Unknown"


def first_genre(value: object) -> str:
    genres = clean_genres(value).split(",")
    return genres[0].strip() if genres else "Unknown"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data[REQUIRED_COLUMNS].drop_duplicates()

    data["title"] = data["title"].fillna("Untitled").astype(str).str.strip()
    data["titleType"] = data["titleType"].fillna("movie").astype(str).str.strip()
    data["genres"] = data["genres"].apply(clean_genres)
    data["primaryGenre"] = data["genres"].apply(first_genre)
    data["averageRating"] = pd.to_numeric(data["averageRating"], errors="coerce")
    data["numVotes"] = pd.to_numeric(data["numVotes"], errors="coerce")
    data["year"] = pd.to_numeric(data["year"], errors="coerce")

    data["averageRating"] = data["averageRating"].fillna(data["averageRating"].median()).clip(0, 10)
    data["numVotes"] = data["numVotes"].fillna(0).clip(lower=0)
    data["year"] = data["year"].fillna(data["year"].median()).astype(int)

    numeric = data[["averageRating", "numVotes"]].to_numpy()
    if len(data) > 1 and MinMaxScaler is not None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(numeric)
    elif len(data) > 1:
        mins = numeric.min(axis=0)
        spans = numeric.max(axis=0) - mins
        spans[spans == 0] = 1
        scaled = (numeric - mins) / spans
    else:
        scaled = np.array([[1.0, 1.0]])
    data["ratingNormalized"] = scaled[:, 0]
    data["votesNormalized"] = scaled[:, 1]
    data["decade"] = (data["year"] // 10 * 10).astype(int)
    data["combinedFeatures"] = (
        data["genres"].str.replace(",", " ", regex=False)
        + " "
        + data["titleType"]
        + " rating_"
        + data["averageRating"].round().astype(int).astype(str)
    )

    data = data.sort_values(["averageRating", "numVotes"], ascending=False).reset_index(drop=True)
    return data


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    exploded = df.copy()
    exploded["genre"] = exploded["genres"].str.split(",")
    exploded = exploded.explode("genre")
    exploded["genre"] = exploded["genre"].astype(str).str.strip()
    return exploded[exploded["genre"].ne("")]
