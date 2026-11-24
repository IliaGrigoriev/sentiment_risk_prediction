from data_managment.data_manager import DataManager
from features import FeatureExtractor
from knowledge_graph import RDFExporter, TriplestoreClient
from risk_analysis.risk_analyzer import RiskAnalyzer
from risk_analysis.risk_visualizer import RiskVisualizer
from const import (
    KNOWLEDGE_GRAPH_PATH,
    TRIPLESTORE_GRAPH_STORE_URL,
    TRIPLESTORE_GRAPH_URI,
    TRIPLESTORE_PASSWORD,
    TRIPLESTORE_QUERY_URL,
    TRIPLESTORE_USERNAME,
)


FEATURE_SAMPLE_PERCENT = 0.5  # Approximately 11275 records


# Run sentiment-based vehicle safety risk analysis
# ------------------------------------------------
def routine():
    print("1/6 Cleaning complaints and recall documents...")
    data_manager = DataManager()
    data_manager.load_and_clean_data()

    print("2/6 Matching complaints to potential recall campaigns...")
    data_manager.match_complaints_to_recalls()
    data_manager.build_report()
    data_manager.save_intermediate_data()

    print("3/6 Extracting sentiment and safety features from a sample...")
    feature_extractor = FeatureExtractor(
        sample_percent=FEATURE_SAMPLE_PERCENT,
    )
    print(feature_extractor.get_sample_info())
    feature_extractor.save_features()

    print("4/6 Summarizing vehicle risk and creating demo figures...")
    features            = feature_extractor.get_features()
    risk_analyzer       = RiskAnalyzer()
    risk_summary        = risk_analyzer.build_vehicle_summary(features)
    recall_risk_summary = risk_analyzer.build_recall_summary(features)
    data_manager.save_risk_summaries(risk_summary, recall_risk_summary)
    RiskVisualizer().create_figures(risk_summary, recall_risk_summary)

    print("5/6 Exporting analysis results as RDF triples...")
    rdf_exporter = RDFExporter()
    graph = rdf_exporter.build_graph(
        features,
        risk_summary,
        recall_risk_summary,
    )
    rdf_exporter.save(graph, KNOWLEDGE_GRAPH_PATH)

    if TRIPLESTORE_GRAPH_STORE_URL:
        print("6/6 Uploading RDF triples to the triplestore...")
        TriplestoreClient(
            graph_store_url = TRIPLESTORE_GRAPH_STORE_URL,
            query_url       = TRIPLESTORE_QUERY_URL,
            username        = TRIPLESTORE_USERNAME,
            password        = TRIPLESTORE_PASSWORD,
        ).upload(graph, graph_uri=TRIPLESTORE_GRAPH_URI)
    else:
        print("6/6 Triplestore upload skipped (endpoint not configured).")

    print("Sentiment-based risk analysis completed.")


if __name__ == "__main__":
    routine()
