from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from preprocessing import explode_genres


def dataset_insights(df: pd.DataFrame) -> dict[str, str]:
    genre_data = explode_genres(df)
    decade_scores = df.groupby("decade")["averageRating"].mean()
    highest_genre = genre_data.groupby("genre")["averageRating"].mean().idxmax()
    common_genre = genre_data["genre"].value_counts().idxmax()
    best_decade = int(decade_scores.idxmax())
    worst_decade = int(decade_scores.idxmin())
    most_voted = df.loc[df["numVotes"].idxmax(), "title"]
    top_rated = df.sort_values(["averageRating", "numVotes"], ascending=False).iloc[0]["title"]
    return {
        "Highest rated genre": highest_genre,
        "Most common genre": common_genre,
        "Best decade": f"{best_decade}s",
        "Worst decade": f"{worst_decade}s",
        "Most voted movie": most_voted,
        "Top rated movie": top_rated,
    }


def generate_pdf_report(
    df: pd.DataFrame,
    recommendations: pd.DataFrame | None = None,
    selected_title: str | None = None,
    output_dir: str | Path = "reports",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"movie_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = output_path / filename

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(report_path), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch)
    story = []

    insights = dataset_insights(df)
    top_genre = insights["Most common genre"]
    best_movie = insights["Top rated movie"]

    story.append(Paragraph("AI-Powered Movie Recommendation Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    summary_rows = [
        ["Total Titles", f"{len(df):,}"],
        ["Average Rating", f"{df['averageRating'].mean():.2f}"],
        ["Top Genre", top_genre],
        ["Best Movie", best_movie],
        ["Total Votes", f"{int(df['numVotes'].sum()):,}"],
    ]
    story.append(_table(summary_rows, [2.2 * inch, 4.5 * inch]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Automated Insights", styles["Heading2"]))
    story.append(_table([[key, value] for key, value in insights.items()], [2.4 * inch, 4.3 * inch]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Top Rated Movies", styles["Heading2"]))
    top = df.nlargest(10, ["averageRating", "numVotes"])[["title", "averageRating", "numVotes", "genres"]]
    story.append(_table([top.columns.tolist()] + top.astype(str).values.tolist(), [2.4 * inch, 1.0 * inch, 1.1 * inch, 2.2 * inch], header=True))

    story.append(PageBreak())
    story.append(Paragraph("Visual Summary", styles["Heading2"]))
    story.append(Paragraph("Rating Distribution", styles["Heading3"]))
    story.append(_rating_distribution_chart(df))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Genre Popularity", styles["Heading3"]))
    story.append(_genre_popularity_chart(df))

    story.append(PageBreak())
    story.append(Paragraph("Release Trend", styles["Heading3"]))
    story.append(_release_trend_chart(df))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Top Rated Movie Ratings", styles["Heading3"]))
    story.append(_top_rated_chart(df))

    if recommendations is not None and not recommendations.empty:
        story.append(PageBreak())
        story.append(Paragraph("Recommendation Summary", styles["Heading2"]))
        story.append(Paragraph(f"Selected title: {selected_title or 'N/A'}", styles["Normal"]))
        recs = recommendations[["title", "similarityScore", "averageRating", "genres"]].copy()
        recs["similarityScore"] = recs["similarityScore"].map(lambda value: f"{value:.3f}")
        story.append(_table([recs.columns.tolist()] + recs.astype(str).values.tolist(), [2.5 * inch, 1.2 * inch, 1.1 * inch, 2.0 * inch], header=True))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Methodology", styles["Heading2"]))
    story.append(
        Paragraph(
            "Recommendations use TF-IDF vectors built from genre, title type, and rating features. "
            "Cosine similarity ranks titles with the closest content profile, then user filters are applied.",
            styles["BodyText"],
        )
    )

    doc.build(story)
    return report_path


def _table(rows: list[list[str]], widths: list[float], header: bool = False) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20283a") if header else colors.HexColor("#151a24")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa4b2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold" if header else "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
    ]
    table.setStyle(TableStyle(style))
    return table


def _rating_distribution_chart(df: pd.DataFrame) -> Drawing:
    bins = list(range(0, 11))
    counts = []
    for start in bins[:-1]:
        counts.append(int(df["averageRating"].between(start, start + 1, inclusive="left").sum()))
    drawing = Drawing(480, 210)
    chart = VerticalBarChart()
    chart.x = 45
    chart.y = 35
    chart.height = 135
    chart.width = 395
    chart.data = [counts]
    chart.categoryAxis.categoryNames = [f"{i}-{i + 1}" for i in bins[:-1]]
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(counts) + 2
    chart.valueAxis.valueStep = max(1, round((max(counts) + 2) / 5))
    chart.bars[0].fillColor = colors.HexColor("#47c2ff")
    drawing.add(chart)
    drawing.add(String(45, 185, f"Average rating: {df['averageRating'].mean():.2f} | Median rating: {df['averageRating'].median():.2f}", fontSize=9))
    return drawing


def _genre_popularity_chart(df: pd.DataFrame) -> Drawing:
    counts = explode_genres(df)["genre"].value_counts().head(10).sort_values()
    drawing = Drawing(480, 250)
    chart = HorizontalBarChart()
    chart.x = 110
    chart.y = 25
    chart.height = 190
    chart.width = 315
    chart.data = [counts.tolist()]
    chart.categoryAxis.categoryNames = counts.index.tolist()
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = int(counts.max()) + 2
    chart.valueAxis.valueStep = max(1, round((int(counts.max()) + 2) / 5))
    chart.bars[0].fillColor = colors.HexColor("#7ee787")
    drawing.add(chart)
    return drawing


def _release_trend_chart(df: pd.DataFrame) -> Drawing:
    yearly = df.groupby("year").size().sort_index()
    values = list(zip(yearly.index.astype(int).tolist(), yearly.astype(int).tolist()))
    drawing = Drawing(480, 230)
    chart = LinePlot()
    chart.x = 45
    chart.y = 35
    chart.height = 145
    chart.width = 390
    chart.data = [values]
    chart.lines[0].strokeColor = colors.HexColor("#ffb86b")
    chart.lines[0].strokeWidth = 2
    chart.xValueAxis.valueMin = min(yearly.index)
    chart.xValueAxis.valueMax = max(yearly.index)
    chart.xValueAxis.valueStep = max(1, round((max(yearly.index) - min(yearly.index)) / 6))
    chart.yValueAxis.valueMin = 0
    chart.yValueAxis.valueMax = int(yearly.max()) + 2
    chart.yValueAxis.valueStep = max(1, round((int(yearly.max()) + 2) / 5))
    drawing.add(chart)
    drawing.add(String(45, 190, "Number of movies and TV shows released per year", fontSize=9))
    return drawing


def _top_rated_chart(df: pd.DataFrame) -> Drawing:
    top = df.nlargest(8, ["averageRating", "numVotes"]).sort_values("averageRating")
    drawing = Drawing(480, 250)
    chart = HorizontalBarChart()
    chart.x = 145
    chart.y = 25
    chart.height = 190
    chart.width = 280
    chart.data = [top["averageRating"].round(2).tolist()]
    chart.categoryAxis.categoryNames = [title[:22] for title in top["title"]]
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 10
    chart.valueAxis.valueStep = 2
    chart.bars[0].fillColor = colors.HexColor("#c792ea")
    drawing.add(chart)
    return drawing
