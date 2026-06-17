from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from preprocessing import REQUIRED_COLUMNS, explode_genres, load_data, preprocess_data
from recommendation import MovieRecommender
from report_generator import dataset_insights, generate_pdf_report
import visualizations as viz


BASE_DIR = Path(__file__).parent
DEFAULT_DATA = BASE_DIR / "data" / "imdb_movies.csv"
REPORT_DIR = BASE_DIR / "reports"


st.set_page_config(page_title="AI Movie Recommender", page_icon="film", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #0f141d; color: #f4f7fb; }
    [data-testid="stSidebar"] { background: #151b27; }
    .metric-card {
        background: #182232;
        border: 1px solid #2b3548;
        border-radius: 8px;
        padding: 16px;
        min-height: 112px;
    }
    .metric-card span { color: #9fb0c8; font-size: 0.82rem; text-transform: uppercase; }
    .metric-card strong { display:block; margin-top: 8px; font-size: 1.55rem; color: #ffffff; }
    .insight-card {
        background: #121a27;
        border-left: 4px solid #47c2ff;
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 88px;
    }
    .insight-card span { color: #9fb0c8; font-size: 0.8rem; }
    .insight-card strong { color: #ffffff; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        return preprocess_data(raw)
    return load_data(DEFAULT_DATA)


@st.cache_resource(show_spinner=False)
def get_recommender(data: pd.DataFrame) -> MovieRecommender:
    return MovieRecommender(data)


def metric_card(label: str, value: str) -> None:
    st.markdown(f"<div class='metric-card'><span>{label}</span><strong>{value}</strong></div>", unsafe_allow_html=True)


def insight_card(label: str, value: str) -> None:
    st.markdown(f"<div class='insight-card'><span>{label}</span><br><strong>{value}</strong></div>", unsafe_allow_html=True)


def main() -> None:
    st.title("AI-Powered Movie Recommendation and Analytics System")
    st.caption("Content-based recommendations, IMDb-style analytics, automated insights, and downloadable reporting.")

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload imdb_movies.csv", type=["csv"])
        st.caption(f"Required columns: {', '.join(REQUIRED_COLUMNS)}")

    try:
        df = get_data(uploaded_file)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

    recommender = get_recommender(df)
    genres = ["All"] + sorted(explode_genres(df)["genre"].unique().tolist())
    title_types = ["All"] + sorted(df["titleType"].dropna().unique().tolist())

    with st.sidebar:
        search = st.text_input("Movie Search", value=df.iloc[0]["title"])
        matches = recommender.search_titles(search, 12)
        selected_title = st.selectbox("Autocomplete Search", matches if matches else df["title"].tolist())
        genre_filter = st.selectbox("Genre Filter", genres)
        year_range = st.slider("Year Range", int(df["year"].min()), int(df["year"].max()), (int(df["year"].min()), int(df["year"].max())))
        rating_filter = st.slider("Rating Threshold", 0.0, 10.0, 6.0, 0.1)
        title_type = st.selectbox("Title Type", title_types)
        top_n = st.slider("Top N Recommendations", 5, 25, 10)

    filtered = df[
        df["year"].between(year_range[0], year_range[1])
        & df["averageRating"].ge(rating_filter)
        & (df["genres"].str.contains(genre_filter, case=False, regex=False) if genre_filter != "All" else True)
        & (df["titleType"].eq(title_type) if title_type != "All" else True)
    ]

    recommendations = recommender.recommend(
        selected_title,
        top_n=top_n,
        genre=genre_filter,
        year_range=year_range,
        min_rating=rating_filter,
        title_type=title_type,
    )
    selected_movie = df[df["title"].eq(selected_title)].iloc[0]

    tab_overview, tab_recs, tab_analytics, tab_trends, tab_report = st.tabs(
        ["Dataset Overview", "Recommendation Engine", "Analytics Dashboard", "Trend Analysis", "Report Generation"]
    )

    with tab_overview:
        st.subheader("Dataset Summary")
        metric_values = [
            ("Total Movies", f"{df[df['titleType'].eq('movie')].shape[0]:,}"),
            ("Total TV Shows", f"{df[df['titleType'].str.contains('tv', case=False, na=False)].shape[0]:,}"),
            ("Average Rating", f"{df['averageRating'].mean():.2f}"),
            ("Total Votes", f"{int(df['numVotes'].sum()):,}"),
            ("Number of Genres", f"{explode_genres(df)['genre'].nunique():,}"),
            ("Latest Release Year", str(int(df["year"].max()))),
        ]
        cols = st.columns(3)
        for index, (label, value) in enumerate(metric_values):
            with cols[index % 3]:
                metric_card(label, value)

        st.subheader("Filtered Dataset")
        st.dataframe(filtered[["title", "titleType", "genres", "averageRating", "numVotes", "year"]], width="stretch")

    with tab_recs:
        st.subheader("Recommendation Engine")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("TF-IDF Matrix Shape", f"{recommender.tfidf_matrix.shape[0]} x {recommender.tfidf_matrix.shape[1]}")
        col_b.metric("Vocabulary Size", f"{recommender.vocabulary_size:,}")
        col_c.metric("Similarity Matrix Size", f"{recommender.similarity_matrix.shape[0]} x {recommender.similarity_matrix.shape[1]}")
        st.info("The engine vectorizes genres, title type, and rating signals with TF-IDF, then ranks titles by cosine similarity.")

        st.write(f"Selected movie: **{selected_title}**")
        st.dataframe(recommendations, width="stretch")
        if not recommendations.empty:
            col1, col2 = st.columns(2)
            col1.plotly_chart(viz.similarity_chart(recommendations), width="stretch")
            col2.plotly_chart(viz.recommendation_rating_chart(selected_movie, recommendations), width="stretch")
            st.plotly_chart(viz.genre_network(selected_movie, recommendations), width="stretch")

    with tab_analytics:
        st.subheader("Analytics Dashboard")
        col1, col2 = st.columns(2)
        col1.plotly_chart(viz.rating_distribution(filtered), width="stretch")
        col2.plotly_chart(viz.genre_popularity(filtered), width="stretch")
        col3, col4 = st.columns(2)
        col3.plotly_chart(viz.genre_average_rating(filtered), width="stretch")
        col4.plotly_chart(viz.most_popular(filtered), width="stretch")
        st.plotly_chart(viz.rating_vs_votes(filtered), width="stretch")
        st.plotly_chart(viz.correlation_heatmap(filtered), width="stretch")
        st.subheader("Top Rated Movies")
        st.dataframe(df.nlargest(20, ["averageRating", "numVotes"])[["title", "averageRating", "numVotes", "genres"]], width="stretch")

    with tab_trends:
        st.subheader("Trend Analysis")
        st.plotly_chart(viz.release_trend(filtered), width="stretch")
        col1, col2 = st.columns(2)
        col1.plotly_chart(viz.decade_analysis(filtered), width="stretch")
        col2.plotly_chart(viz.genre_treemap(filtered), width="stretch")
        st.plotly_chart(viz.top_genres_by_decade(filtered), width="stretch")
        col3, col4 = st.columns(2)
        col3.plotly_chart(viz.rating_trend(filtered), width="stretch")
        col4.plotly_chart(viz.voting_trend(filtered), width="stretch")

        st.subheader("Automated Insights")
        insight_cols = st.columns(3)
        for index, (label, value) in enumerate(dataset_insights(df).items()):
            with insight_cols[index % 3]:
                insight_card(label, value)

    with tab_report:
        st.subheader("PDF Report Generator")
        st.write("Generate a downloadable PDF report with summary metrics, top titles, methodology, and current recommendation results.")
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Building report..."):
                report_path = generate_pdf_report(df, recommendations, selected_title, REPORT_DIR)
                st.success(f"Report saved automatically to {report_path}")
                st.download_button(
                    "Download PDF Report",
                    data=report_path.read_bytes(),
                    file_name=report_path.name,
                    mime="application/pdf",
                )


if __name__ == "__main__":
    main()
