import json


class DataReporter:

    # Build a quality and matching report
    # -----------------------------------
    def build(self, complaints, recalls, matched, source_recall_count, dedupe_key):
        report = {
            "complaints": {
                "records": len(complaints),
                "available_narratives": int(
                    complaints["complaint_narrative"].notna().sum()
                ),
                "missing_values": self._missing_values(
                    complaints,
                    ["manufacturer", "make", "model", "model_year", "complaint_date"],
                ),
                "complaint_date_range": self._date_range(
                    complaints,
                    "complaint_date",
                ),
            },
            "recalls": {
                "source_records": source_recall_count,
                "cleaned_records": len(recalls),
                "duplicates_removed": source_recall_count - len(recalls),
                "distinct_campaigns": int(recalls["nhtsa_id"].nunique(dropna=True)),
                "deduplication_key": dedupe_key,
                "missing_values": self._missing_values(
                    recalls,
                    ["nhtsa_id", "make", "model", "model_year", "summary"],
                ),
            },
            "matching": {
                "matched_complaints": int(matched["has_matching_recall"].sum()),
                "unmatched_complaints": int((~matched["has_matching_recall"]).sum()),
                "match_rate": float(matched["has_matching_recall"].mean()),
            },
        }
        print(json.dumps(report, indent=2))
        return report

    # Count missing values in selected columns
    # ----------------------------------------
    def _missing_values(self, data, columns):
        return {
            column: int(value)
            for column, value in data[columns].isna().sum().items()
        }

    # Calculate the observed range of a date column
    # ---------------------------------------------
    def _date_range(self, data, column):
        values = data[column].dropna()
        if values.empty:
            return {"minimum": None, "maximum": None}
        return {
            "minimum": values.min().date().isoformat(),
            "maximum": values.max().date().isoformat(),
        }
