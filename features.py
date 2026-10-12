import os

import pandas as pd

from const import (
    COMPLAINT_FEATURES_PATH,
    FEATURE_INFERENCE_CHUNK_SIZE,
    FEATURE_SAMPLE_RANDOM_STATE,
    INTERM_RES,
    MATCHED_COMPLAINTS_PATH,
)
from sentiment_estimator import SentimentEstimator


class FeatureExtractor:

    # Safety severity ordinal scale
    # 0: No severe outcome recorded
    # 1: Vehicle required towing
    # 2: Crash reported without fire, injury, or death
    # 3: Fire reported without injury or death
    # 4: Injury or medical attention reported
    # 5: Death reported
    # ------------------------------------------
    _SAFETY_SEVERITY_RULES = [
        ("VEHICLES_TOWED_YN", "boolean", 1),
        ("CRASH", "boolean", 2),
        ("FIRE", "boolean", 3),
        ("MEDICAL_ATTN", "boolean", 4),
        ("INJURED", "numeric", 4),
        ("DEATHS", "numeric", 5),
    ]

    _FEATURE_COLUMNS = [
        "CMPLID",
        "ODINO",
        "manufacturer",
        "make",
        "model",
        "model_year",
        "complaint_narrative",
        "CRASH",
        "FIRE",
        "INJURED",
        "DEATHS",
        "MEDICAL_ATTN",
        "VEHICLES_TOWED_YN",
        "has_matching_recall",
        "matching_recall_count",
        "matching_recall_ids",
    ]

    # Load and sample the complaint fields required for feature extraction
    # --------------------------------------------------------------------
    def __init__(
        self,
        sample_percent,
        data_path=MATCHED_COMPLAINTS_PATH,
    ):
        if not 0 < sample_percent <= 100:
            raise ValueError("sample_percent must be greater than 0 and at most 100")

        self._data_path = data_path
        self._data = pd.read_parquet(data_path, columns=self._FEATURE_COLUMNS)
        sample_size = max(
            1,
            round(len(self._data) * sample_percent / 100),
        )
        self._data = self._data.sample(
            n=sample_size,
            random_state=FEATURE_SAMPLE_RANDOM_STATE,
        ).sort_index()
        self._sample_percent = sample_percent
        self._sentiment_estimator = None
        self._features = None

    # Lazily initialize the transformer sentiment estimator
    # -----------------------------------------------------
    def _get_sentiment_estimator(self):
        if self._sentiment_estimator is None:
            self._sentiment_estimator = SentimentEstimator()

        return self._sentiment_estimator

    # Extract negative sentiment probabilities from complaint narratives
    # ------------------------------------------------------------------
    def extract_sentiment_score(self):
        scores = pd.Series(pd.NA, index=self._data.index, dtype="Float32")
        narratives = self._data["complaint_narrative"]
        valid_indexes = narratives[narratives.notna() & narratives.str.strip().ne("")].index
        estimator = self._get_sentiment_estimator()

        for start in range(0, len(valid_indexes), FEATURE_INFERENCE_CHUNK_SIZE):
            chunk_indexes = valid_indexes[start:start + FEATURE_INFERENCE_CHUNK_SIZE]
            texts = narratives.loc[chunk_indexes].tolist()
            results = estimator.get_sentiment(texts)
            negative_scores = [
                result["probabilities"]["negative"]
                for result in results
            ]
            scores.loc[chunk_indexes] = negative_scores

        return scores

    # Extract an independent safety-severity score from reported outcomes
    # -------------------------------------------------------------------
    def extract_safety_severity_score(self):
        scores = pd.Series(0, index=self._data.index, dtype="Int8")

        for column, rule_type, severity in self._SAFETY_SEVERITY_RULES:
            if rule_type == "numeric":
                condition = pd.to_numeric(
                    self._data[column], errors="coerce"
                ).gt(0)
            else:
                condition = self._data[column].eq("Y")

            scores.loc[condition.fillna(False)] = scores.loc[
                condition.fillna(False)
            ].clip(lower=severity)

        return scores

    # Extract sentiment and safety severity as independent features
    # -------------------------------------------------------------
    def extract_features(self):
        identifier_columns = [
            column
            for column in [
                "CMPLID",
                "ODINO",
                "manufacturer",
                "make",
                "model",
                "model_year",
                "has_matching_recall",
                "matching_recall_count",
                "matching_recall_ids",
            ]
            if column in self._data.columns
        ]
        self._features = self._data[identifier_columns].copy()
        self._features["sentiment_score"] = self.extract_sentiment_score()
        self._features["safety_severity_score"] = (
            self.extract_safety_severity_score()
        )

        return self._features

    # Save the extracted features in the intermediate data folder
    # -----------------------------------------------------------
    def save_features(self, output_path=COMPLAINT_FEATURES_PATH):
        if self._features is None:
            self.extract_features()

        os.makedirs(INTERM_RES, exist_ok=True)
        self._features.to_parquet(output_path, index=False)

    # Return the extracted feature dataset
    # ------------------------------------
    def get_features(self):
        if self._features is None:
            return self.extract_features()

        return self._features

    # Return the percentage and number of sampled complaints
    # ------------------------------------------------------
    def get_sample_info(self):
        return {
            "sample_percent": self._sample_percent,
            "sample_records": len(self._data),
        }
