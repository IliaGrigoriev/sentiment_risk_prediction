import pandas as pd


class DataMatcher:

    _MATCH_COLUMNS = ["make", "model", "model_year"]

    # Collapse recall documents into distinct campaign-vehicle records
    # ----------------------------------------------------------------
    def build_campaign_vehicles(self, recalls):
        usable_recalls = recalls.dropna(subset=self._MATCH_COLUMNS + ["nhtsa_id"])
        campaign_vehicles = (
            usable_recalls.groupby(
                ["nhtsa_id"] + self._MATCH_COLUMNS,
                as_index=False,
                dropna=False,
            )
            .agg(
                recall_summary=("summary", "first"),
                recall_document_count=("document_name", "nunique"),
            )
        )
        return campaign_vehicles

    # Match complaints to all potential campaigns for the same vehicle
    # ----------------------------------------------------------------
    def match_complaints_to_recalls(self, complaints, campaign_vehicles):
        vehicle_recalls = (
            campaign_vehicles.groupby(self._MATCH_COLUMNS, as_index=False)
            .agg(
                matching_recall_count=("nhtsa_id", "nunique"),
                matching_recall_ids=(
                    "nhtsa_id",
                    lambda values: "|".join(sorted(set(values.dropna()))),
                ),
            )
        )

        matched = complaints.merge(
            vehicle_recalls,
            how="left",
            on=self._MATCH_COLUMNS,
            validate="many_to_one",
        )
        matched["matching_recall_count"] = (
            matched["matching_recall_count"].fillna(0).astype("Int32")
        )
        matched["matching_recall_ids"] = matched[
            "matching_recall_ids"
        ].fillna("")
        matched["has_matching_recall"] = matched["matching_recall_count"].gt(0)
        return matched

    # Return the columns used for vehicle-level matching
    # --------------------------------------------------
    def match_columns(self):
        return self._MATCH_COLUMNS.copy()
