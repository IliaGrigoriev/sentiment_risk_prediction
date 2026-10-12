from data_managment.complaint_loader import ComplaintLoader
from data_managment.data_cleaner import DataCleaner
from data_managment.data_matcher import DataMatcher
from data_managment.data_reporter import DataReporter
from data_managment.data_store import DataStore
from data_managment.recall_loader import RecallLoader


class DataManager:

    # Initialize data-management collaborators without processing data
    # ---------------------------------------------------------------
    def __init__(self):
        self._complaint_loader = ComplaintLoader()
        self._recall_loader = RecallLoader()
        self._cleaner = DataCleaner()
        self._matcher = DataMatcher()
        self._reporter = DataReporter()
        self._store = DataStore()

        self._complaints_data = None
        self._recalls_data = None
        self._campaign_vehicles = None
        self._matched_complaints = None
        self._report = None
        self._source_recall_count = 0

    # Load and clean complaint and recall source data
    # -----------------------------------------------
    def load_and_clean_data(self):
        complaints = self._complaint_loader.load()
        recalls = self._recall_loader.load()
        self._source_recall_count = len(recalls)

        self._complaints_data = self._cleaner.clean_complaints(complaints)
        self._recalls_data = self._cleaner.clean_recalls(recalls)
        return self._complaints_data, self._recalls_data

    # Prepare campaign-level recalls before complaint feature extraction
    # ------------------------------------------------------------------
    def match_complaints_to_recalls(self):
        self._require_prepared_data()
        self._campaign_vehicles = self._matcher.build_campaign_vehicles(
            self._recalls_data
        )
        self._matched_complaints = self._matcher.match_complaints_to_recalls(
            self._complaints_data,
            self._campaign_vehicles,
        )
        return self._matched_complaints

    # Build the data-quality and pre-feature matching report
    # ------------------------------------------------------
    def build_report(self):
        if self._matched_complaints is None:
            raise RuntimeError("Match complaints before building the report")

        self._report = self._reporter.build(
            self._complaints_data,
            self._recalls_data,
            self._matched_complaints,
            self._source_recall_count,
            self._cleaner.recall_deduplication_columns(),
        )
        return self._report

    # Save all feature-independent intermediate datasets
    # --------------------------------------------------
    def save_intermediate_data(self):
        if self._matched_complaints is None or self._report is None:
            raise RuntimeError("Match and report data before saving it")

        self._store.save_intermediate(
            self._complaints_data,
            self._recalls_data,
            self._matched_complaints,
            self._report,
        )

    # Save feature-dependent analysis results without analyzing them
    # --------------------------------------------------------------
    def save_risk_summaries(self, vehicle_risk_summary, recall_risk_summary):
        self._store.save_risk_summaries(
            vehicle_risk_summary,
            recall_risk_summary,
        )

    # Ensure cleaned source data exists before dependent operations
    # -------------------------------------------------------------
    def _require_prepared_data(self):
        if self._complaints_data is None or self._recalls_data is None:
            raise RuntimeError("Load and clean data before matching it")

    # Return the matched complaint dataset
    # ------------------------------------
    def get_matched_complaints(self):
        return self._matched_complaints

