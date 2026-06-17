# AI-Powered Movie Recommendation and Analytics System

Production-style Streamlit project built from the PDF specification. It uses an IMDb-style dataset to provide content-based recommendations, analytics dashboards, automated insights, and downloadable PDF reports.

## Features

- Dark themed Streamlit dashboard
- CSV upload support for IMDb-style data
- Data preprocessing for missing values, duplicates, genres, normalized ratings/votes, and decade extraction
- TF-IDF plus cosine similarity recommendation engine
- Partial search, autocomplete, and fuzzy matching with RapidFuzz
- Filters for genre, year range, rating threshold, title type, and top N results
- Dataset KPI cards
- 10+ Plotly visualizations
- Automated insight cards
- Recommendation comparison charts
- PDF report generation with automatic save and download

## Project Structure

```text
movie-recommendation-system/
|-- app.py
|-- recommendation.py
|-- preprocessing.py
|-- visualizations.py
|-- report_generator.py
|-- requirements.txt
|-- data/
|   `-- imdb_movies.csv
|-- assets/
|-- reports/
`-- README.md
```

## Dataset Columns

The app expects a CSV with these columns:

- `title`
- `titleType`
- `genres`
- `averageRating`
- `numVotes`
- `year`

A sample dataset is included at `data/imdb_movies.csv`. Replace it with your full IMDb export or upload a CSV in the app sidebar.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Methodology

The recommender builds a combined content feature from genres, title type, and rounded rating signal. TF-IDF vectorization converts those features into a sparse matrix, and cosine similarity ranks titles by content proximity. Sidebar filters are applied after ranking to keep recommendations relevant to the user selected constraints.

## PDF Reports

Reports are saved automatically in `reports/` and are also available from the Streamlit download button after generation.
