import os

import plotly.express as px

from const import FIGS_PATH


class RiskVisualizer:

    # Initialize the destination for interactive demo figures
    # -------------------------------------------------------
    def __init__(self, figures_path=FIGS_PATH):
        self._figures_path = figures_path

    # Create the sentiment-versus-safety vehicle risk map
    # ---------------------------------------------------
    def create_risk_map(self, risk_summary):
        os.makedirs(self._figures_path, exist_ok=True)
        figure_data = risk_summary.dropna(
            subset=["mean_negative_sentiment", "mean_safety_severity"]
        ).copy()
        figure_data["vehicle"] = (
            figure_data["make"].astype(str)
            + " "
            + figure_data["model"].astype(str)
            + " "
            + figure_data["model_year"].astype(str)
        )

        figure = px.scatter(
            figure_data,
            x="mean_negative_sentiment",
            y="mean_safety_severity",
            size="complaint_count",
            color="has_matching_recall",
            hover_name="vehicle",
            hover_data=[
                "complaint_count",
                "max_safety_severity",
                "matching_recall_count",
            ],
            labels={
                "mean_negative_sentiment": "Mean negative sentiment",
                "mean_safety_severity": "Mean safety severity (0–5)",
                "has_matching_recall": "Potential recall match",
            },
            title="Consumer Sentiment and Reported Safety Severity",
        )
        figure.write_html(
            os.path.join(self._figures_path, "sentiment_safety_risk_map.html"),
            include_plotlyjs="cdn",
        )

    # Create a ranking of severe vehicles without matching recalls
    # ------------------------------------------------------------
    def create_emerging_signals(self, risk_summary):
        os.makedirs(self._figures_path, exist_ok=True)
        signals = risk_summary[
            risk_summary["potential_emerging_signal"]
        ].head(20).copy()
        if signals.empty:
            return

        signals["vehicle"] = (
            signals["make"].astype(str)
            + " "
            + signals["model"].astype(str)
            + " "
            + signals["model_year"].astype(str)
        )
        figure = px.bar(
            signals.sort_values("severe_complaint_count"),
            x="severe_complaint_count",
            y="vehicle",
            orientation="h",
            color="mean_negative_sentiment",
            hover_data=["complaint_count", "max_safety_severity"],
            labels={
                "severe_complaint_count": "Severe complaints",
                "vehicle": "Vehicle",
                "mean_negative_sentiment": "Negative sentiment",
            },
            title="Potential Emerging Signals Without Matching Recalls",
        )
        figure.write_html(
            os.path.join(self._figures_path, "potential_emerging_signals.html"),
            include_plotlyjs="cdn",
        )

    # Create complaint volume and severity by recall campaign
    # -------------------------------------------------------
    def create_recall_volume(self, recall_risk_summary):
        os.makedirs(self._figures_path, exist_ok=True)
        if recall_risk_summary.empty:
            return

        campaigns = recall_risk_summary.head(20).sort_values("complaint_count")
        figure = px.bar(
            campaigns,
            x="complaint_count",
            y="nhtsa_id",
            orientation="h",
            color="mean_safety_severity",
            hover_data=[
                "severe_complaint_count",
                "mean_negative_sentiment",
                "max_safety_severity",
            ],
            labels={
                "complaint_count": "Potentially associated complaints",
                "nhtsa_id": "NHTSA vehicle recall campaign ID",
                "mean_safety_severity": "Mean safety severity",
            },
            title=(
                "Complaint Volume Associated with Recall Campaigns"
                "<br><sup>Campaign ID example: 18V268 — 18 = 2018, "
                "V = vehicle recall (not volts), 268 = campaign sequence.</sup>"
            ),
        )
        figure.update_layout(margin={"t": 110})
        figure.write_html(
            os.path.join(self._figures_path, "recall_complaint_volume.html"),
            include_plotlyjs="cdn",
        )

    # Create every demo figure
    # ------------------------
    def create_figures(self, risk_summary, recall_risk_summary):
        self.create_risk_map(risk_summary)
        self.create_emerging_signals(risk_summary)
        self.create_recall_volume(recall_risk_summary)
