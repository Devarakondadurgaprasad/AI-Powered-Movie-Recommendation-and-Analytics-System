from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import explode_genres


PLOT_TEMPLATE = "plotly_dark"


def rating_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="averageRating", nbins=24, marginal="box", template=PLOT_TEMPLATE)
    fig.update_layout(title="Rating Distribution", xaxis_title="Rating", yaxis_title="Titles")
    return fig


def genre_popularity(df: pd.DataFrame, limit: int = 15) -> go.Figure:
    counts = explode_genres(df)["genre"].value_counts().head(limit).sort_values()
    fig = px.bar(counts, x=counts.values, y=counts.index, orientation="h", template=PLOT_TEMPLATE)
    fig.update_layout(title="Top Genres by Title Count", xaxis_title="Titles", yaxis_title="")
    return fig


def genre_average_rating(df: pd.DataFrame) -> go.Figure:
    ratings = explode_genres(df).groupby("genre", as_index=False)["averageRating"].mean()
    ratings = ratings.sort_values("averageRating", ascending=False).head(15)
    fig = px.bar(ratings, x="genre", y="averageRating", template=PLOT_TEMPLATE)
    fig.update_layout(title="Average Rating by Genre", xaxis_title="", yaxis_title="Average Rating")
    return fig


def release_trend(df: pd.DataFrame) -> go.Figure:
    yearly = df.groupby("year", as_index=False).size()
    fig = px.line(yearly, x="year", y="size", markers=True, template=PLOT_TEMPLATE)
    fig.update_layout(title="Movies and Shows Released Per Year", xaxis_title="Year", yaxis_title="Titles")
    return fig


def most_popular(df: pd.DataFrame, limit: int = 12) -> go.Figure:
    data = df.nlargest(limit, "numVotes").sort_values("numVotes")
    fig = px.bar(data, x="numVotes", y="title", color="averageRating", orientation="h", template=PLOT_TEMPLATE)
    fig.update_layout(title="Most Popular Titles", xaxis_title="Votes", yaxis_title="")
    return fig


def rating_vs_votes(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="numVotes",
        y="averageRating",
        color="primaryGenre",
        size="averageRating",
        hover_name="title",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(title="Rating vs Votes", xaxis_title="Votes", yaxis_title="Rating")
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df[["averageRating", "numVotes", "year"]].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", template=PLOT_TEMPLATE, color_continuous_scale="RdBu_r")
    fig.update_layout(title="Correlation Matrix")
    return fig


def genre_treemap(df: pd.DataFrame) -> go.Figure:
    counts = explode_genres(df).groupby("genre", as_index=False).size()
    fig = px.treemap(counts, path=["genre"], values="size", template=PLOT_TEMPLATE)
    fig.update_layout(title="Genre Treemap")
    return fig


def decade_analysis(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("decade", as_index=False).agg(averageRating=("averageRating", "mean"), titles=("title", "count"))
    fig = go.Figure()
    fig.add_bar(x=data["decade"], y=data["titles"], name="Title Count", yaxis="y")
    fig.add_scatter(x=data["decade"], y=data["averageRating"], name="Average Rating", yaxis="y2", mode="lines+markers")
    fig.update_layout(
        title="Decade Analysis",
        template=PLOT_TEMPLATE,
        yaxis=dict(title="Titles"),
        yaxis2=dict(title="Average Rating", overlaying="y", side="right", range=[0, 10]),
    )
    return fig


def top_genres_by_decade(df: pd.DataFrame) -> go.Figure:
    data = explode_genres(df)
    grouped = data.groupby(["decade", "genre"], as_index=False).size()
    top = grouped.groupby("genre")["size"].sum().nlargest(8).index
    grouped = grouped[grouped["genre"].isin(top)]
    fig = px.bar(grouped, x="decade", y="size", color="genre", template=PLOT_TEMPLATE)
    fig.update_layout(title="Top Genres by Decade", xaxis_title="Decade", yaxis_title="Titles")
    return fig


def rating_trend(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("year", as_index=False)["averageRating"].mean()
    fig = px.line(data, x="year", y="averageRating", markers=True, template=PLOT_TEMPLATE)
    fig.update_layout(title="Average Rating Over Years", yaxis_title="Average Rating")
    return fig


def voting_trend(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("year", as_index=False)["numVotes"].mean()
    fig = px.line(data, x="year", y="numVotes", markers=True, template=PLOT_TEMPLATE)
    fig.update_layout(title="Average Votes Over Years", yaxis_title="Average Votes")
    return fig


def similarity_chart(recommendations: pd.DataFrame) -> go.Figure:
    data = recommendations.sort_values("similarityScore")
    fig = px.bar(data, x="similarityScore", y="title", orientation="h", template=PLOT_TEMPLATE)
    fig.update_layout(title="Similarity Scores", xaxis_title="Cosine Similarity", yaxis_title="")
    return fig


def recommendation_rating_chart(selected: pd.Series, recommendations: pd.DataFrame) -> go.Figure:
    data = pd.concat(
        [
            pd.DataFrame({"title": [selected["title"]], "averageRating": [selected["averageRating"]], "kind": ["Selected"]}),
            recommendations.assign(kind="Recommended")[["title", "averageRating", "kind"]],
        ]
    )
    fig = px.bar(data, x="title", y="averageRating", color="kind", template=PLOT_TEMPLATE)
    fig.update_layout(title="Rating Comparison", xaxis_title="", yaxis_title="Rating")
    return fig


def genre_network(selected: pd.Series, recommendations: pd.DataFrame) -> go.Figure:
    nodes = [selected["title"]] + recommendations["title"].head(8).tolist()
    scores = [1.0] + recommendations["similarityScore"].head(8).tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(nodes))),
            y=scores,
            mode="markers+text+lines",
            text=nodes,
            textposition="top center",
            marker=dict(size=[28] + [18] * (len(nodes) - 1), color=scores, colorscale="Viridis"),
        )
    )
    fig.update_layout(title="Genre Similarity Network", template=PLOT_TEMPLATE, xaxis_visible=False, yaxis_title="Similarity")
    return fig
