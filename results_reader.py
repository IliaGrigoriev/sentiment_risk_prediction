from pathlib import Path

import pandas as pd

from const import REPO_PATH, VEHICLE_RISK_SUMMARY_PATH


class ResultsReader:
    _VEHICLE_COLUMNS = ["make", "model", "model_year"]
    _RISK_COLUMNS = [
        "severe_complaint_rate",
        "mean_safety_severity",
        "max_safety_severity",
        "mean_negative_sentiment",
    ]

    def save_top_vehicles(
        self,
        summary_path=VEHICLE_RISK_SUMMARY_PATH,
        output_path=Path(REPO_PATH) / "top_50_vehicles.csv",
    ):
        summary = pd.read_parquet(summary_path)
        vehicles = summary.dropna(subset=self._VEHICLE_COLUMNS)
        vehicles = vehicles.loc[vehicles["model_year"].ne(9999)]

        ranked = vehicles.sort_values(
            self._RISK_COLUMNS + self._VEHICLE_COLUMNS,
            na_position="last",
        )
        # If manufacturer labels split a vehicle group, retain its higher-risk row.
        ranked = ranked.drop_duplicates(subset=self._VEHICLE_COLUMNS, keep="last")
        top_vehicles = ranked[
            self._VEHICLE_COLUMNS + ["complaint_count"] + self._RISK_COLUMNS
        ].head(50).copy()
        top_vehicles.insert(0, "rank", range(1, len(top_vehicles) + 1))
        top_vehicles.to_csv(output_path, index=False)
        return top_vehicles


if __name__ == "__main__":
    results = ResultsReader().save_top_vehicles()
    print(f"Saved {len(results)} vehicle groups to top_50_vehicles.csv.")
