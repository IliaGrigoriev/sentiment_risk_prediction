from data_managment.data_manager import DataManager
from features import FeatureExtractor
from risk_analysis.risk_analyzer import RiskAnalyzer
from risk_analysis.risk_visualizer import RiskVisualizer


FEATURE_SAMPLE_PERCENT = 0.5  # Approximately 11275 records


# Run sentiment-based vehicle safety risk analysis
# ------------------------------------------------
def routine():
    print("1/4 Cleaning complaints and recall documents...")
    data_manager = DataManager()
    data_manager.load_and_clean_data()

    print("2/4 Matching complaints to potential recall campaigns...")
    data_manager.match_complaints_to_recalls()
    data_manager.build_report()
    data_manager.save_intermediate_data()

    print("3/4 Extracting sentiment and safety features from a sample...")
    feature_extractor = FeatureExtractor(
        sample_percent=FEATURE_SAMPLE_PERCENT,
    )
    print(feature_extractor.get_sample_info())
    feature_extractor.save_features()

    print("4/4 Summarizing vehicle risk and creating demo figures...")
    features = feature_extractor.get_features()
    risk_analyzer = RiskAnalyzer()
    risk_summary = risk_analyzer.build_vehicle_summary(features)
    recall_risk_summary = risk_analyzer.build_recall_summary(features)
    data_manager.save_risk_summaries(risk_summary, recall_risk_summary)
    RiskVisualizer().create_figures(risk_summary, recall_risk_summary)

    print("Sentiment-based risk analysis completed.")


if __name__ == "__main__":
    routine()
