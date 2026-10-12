import pandas as pd


class RiskAnalyzer:

    _VEHICLE_COLUMNS = ["manufacturer", "make", "model", "model_year"]

    # Aggregate feature-bearing complaints into vehicle-level risk signals
    # --------------------------------------------------------------------
    def build_vehicle_summary(self, features):
        features = features.copy()
        features["severe_complaint"] = features[
            "safety_severity_score"
        ].ge(3)

        summary = (
            features.groupby(self._VEHICLE_COLUMNS, as_index=False, dropna=False)
            .agg(
                complaint_count=("CMPLID", "size"),
                mean_negative_sentiment=("sentiment_score", "mean"),
                median_negative_sentiment=("sentiment_score", "median"),
                mean_safety_severity=("safety_severity_score", "mean"),
                max_safety_severity=("safety_severity_score", "max"),
                severe_complaint_count=("severe_complaint", "sum"),
                has_matching_recall=("has_matching_recall", "max"),
                matching_recall_count=("matching_recall_count", "max"),
                matching_recall_ids=("matching_recall_ids", "first"),
            )
        )
        summary["severe_complaint_rate"] = (
            summary["severe_complaint_count"] / summary["complaint_count"]
        )
        summary["potential_emerging_signal"] = (
            summary["severe_complaint_count"].gt(0)
            & ~summary["has_matching_recall"]
        )
        return summary.sort_values(
            ["potential_emerging_signal", "severe_complaint_count", "complaint_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    # Aggregate feature-bearing complaints by potential recall campaign
    # -----------------------------------------------------------------
    def build_recall_summary(self, features):
        matched = features[
            features["has_matching_recall"]
            & features["matching_recall_ids"].ne("")
        ].copy()
        if matched.empty:
            return pd.DataFrame(
                columns=[
                    "nhtsa_id",
                    "complaint_count",
                    "mean_negative_sentiment",
                    "mean_safety_severity",
                    "max_safety_severity",
                    "severe_complaint_count",
                    "vehicle_count",
                    "vehicle_examples",
                ]
            )

        matched["nhtsa_id"] = matched["matching_recall_ids"].str.split("|")
        matched = matched.explode("nhtsa_id")
        matched["vehicle"] = (
            matched["make"].astype(str)
            + " "
            + matched["model"].astype(str)
            + " "
            + matched["model_year"].astype(str)
        )
        matched["severe_complaint"] = matched[
            "safety_severity_score"
        ].ge(3)

        return (
            matched.groupby("nhtsa_id", as_index=False)
            .agg(
                complaint_count=("CMPLID", "size"),
                mean_negative_sentiment=("sentiment_score", "mean"),
                mean_safety_severity=("safety_severity_score", "mean"),
                max_safety_severity=("safety_severity_score", "max"),
                severe_complaint_count=("severe_complaint", "sum"),
                vehicle_count=("vehicle", "nunique"),
                vehicle_examples=("vehicle", self._vehicle_examples),
            )
            .sort_values(
                ["severe_complaint_count", "complaint_count"],
                ascending=False,
            )
            .reset_index(drop=True)
        )

    # Format representative vehicles for campaign labels and tooltips
    # ---------------------------------------------------------------
    def _vehicle_examples(self, vehicles):
        unique_vehicles = sorted(set(vehicles.dropna()))
        examples = ", ".join(unique_vehicles[:3])
        remaining = len(unique_vehicles) - 3
        if remaining > 0:
            examples = f"{examples} (+{remaining} more)"
        return examples
