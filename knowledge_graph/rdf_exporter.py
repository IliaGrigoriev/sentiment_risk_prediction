import os
from urllib.parse import quote

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, XSD


class RDFExporter:
    """Convert complaint features and risk summaries into an RDF graph."""

    # Initialize the RDF namespace used for generated resources and properties
    # ------------------------------------------------------------------------
    def __init__(self, base_uri="https://example.org/sentiment-risk/"):
        self._base_uri = base_uri.rstrip("/") + "/"
        self.namespace = Namespace(self._base_uri)

    # Build one RDF graph from complaint features and optional risk summaries
    # ------------------------------------------------------------------------
    def build_graph(self, features, vehicle_summary=None, recall_summary=None):
        graph = Graph()
        graph.bind("sr", self.namespace)

        vehicle_uris = {}
        for _, row in features.iterrows():
            vehicle_uri = self._vehicle_uri(row)
            vehicle_uris[self._vehicle_key(row)] = vehicle_uri
            self._add_vehicle(graph, vehicle_uri, row)
            self._add_complaint(graph, row, vehicle_uri)

        if vehicle_summary is not None:
            for _, row in vehicle_summary.iterrows():
                vehicle_uri = vehicle_uris.get(
                    self._vehicle_key(row), self._vehicle_uri(row)
                )
                self._add_vehicle(graph, vehicle_uri, row)
                self._add_vehicle_summary(graph, vehicle_uri, row)

        if recall_summary is not None:
            for _, row in recall_summary.iterrows():
                self._add_recall_summary(graph, row)

        return graph

    # Serialize an RDF graph to a file in the requested RDF format
    # ------------------------------------------------------------
    def save(self, graph, output_path, rdf_format="turtle"):
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        graph.serialize(destination=output_path, format=rdf_format)

    # Add one complaint and its vehicle and recall relationships to the graph
    # -----------------------------------------------------------------------
    def _add_complaint(self, graph, row, vehicle_uri):
        identifier = self._first_value(row, "CMPLID", "ODINO")
        if identifier is None:
            return

        complaint = self.namespace[f"complaint/{self._encode(identifier)}"]
        graph.add((complaint, RDF.type, self.namespace.Complaint))
        graph.add((complaint, self.namespace.concernsVehicle, vehicle_uri))
        self._add_literal(graph, complaint, "complaintId", row.get("CMPLID"))
        self._add_literal(graph, complaint, "odino", row.get("ODINO"))
        self._add_literal(
            graph, complaint, "negativeSentimentScore", row.get("sentiment_score"), XSD.decimal
        )
        self._add_literal(
            graph, complaint, "safetySeverityScore", row.get("safety_severity_score"), XSD.integer
        )

        recall_ids = row.get("matching_recall_ids")
        if self._has_value(recall_ids):
            for recall_id in str(recall_ids).split("|"):
                if recall_id:
                    recall = self._recall_uri(recall_id)
                    graph.add((recall, RDF.type, self.namespace.RecallCampaign))
                    graph.add((complaint, self.namespace.potentiallyMatchesRecall, recall))

    # Add the identifying properties for one vehicle to the graph
    # ------------------------------------------------------------
    def _add_vehicle(self, graph, vehicle_uri, row):
        graph.add((vehicle_uri, RDF.type, self.namespace.Vehicle))
        self._add_literal(graph, vehicle_uri, "manufacturer", row.get("manufacturer"))
        self._add_literal(graph, vehicle_uri, "make", row.get("make"))
        self._add_literal(graph, vehicle_uri, "model", row.get("model"))
        self._add_literal(graph, vehicle_uri, "modelYear", row.get("model_year"), XSD.integer)

    # Add aggregated risk measurements to an existing vehicle resource
    # -----------------------------------------------------------------
    def _add_vehicle_summary(self, graph, vehicle_uri, row):
        properties = {
            "complaintCount": ("complaint_count", XSD.integer),
            "meanNegativeSentiment": ("mean_negative_sentiment", XSD.decimal),
            "meanSafetySeverity": ("mean_safety_severity", XSD.decimal),
            "maxSafetySeverity": ("max_safety_severity", XSD.integer),
            "severeComplaintCount": ("severe_complaint_count", XSD.integer),
            "severeComplaintRate": ("severe_complaint_rate", XSD.decimal),
            "potentialEmergingSignal": ("potential_emerging_signal", XSD.boolean),
        }
        for predicate, (column, datatype) in properties.items():
            self._add_literal(graph, vehicle_uri, predicate, row.get(column), datatype)

    # Add one recall campaign and its aggregated risk measurements
    # -------------------------------------------------------------
    def _add_recall_summary(self, graph, row):
        recall_id = row.get("nhtsa_id")
        if not self._has_value(recall_id):
            return
        recall = self._recall_uri(recall_id)
        graph.add((recall, RDF.type, self.namespace.RecallCampaign))
        self._add_literal(graph, recall, "nhtsaId", recall_id)
        for predicate, column, datatype in [
            ("associatedComplaintCount", "complaint_count", XSD.integer),
            ("meanNegativeSentiment", "mean_negative_sentiment", XSD.decimal),
            ("meanSafetySeverity", "mean_safety_severity", XSD.decimal),
            ("maxSafetySeverity", "max_safety_severity", XSD.integer),
            ("severeComplaintCount", "severe_complaint_count", XSD.integer),
            ("vehicleCount", "vehicle_count", XSD.integer),
        ]:
            self._add_literal(graph, recall, predicate, row.get(column), datatype)

    # Create a stable vehicle URI from its make, model, and model year
    # ----------------------------------------------------------------
    def _vehicle_uri(self, row):
        parts = [row.get("make"), row.get("model"), row.get("model_year")]
        key = "-".join(self._encode(value) for value in parts)
        return self.namespace[f"vehicle/{key}"]

    # Create a stable recall URI from its NHTSA campaign identifier
    # --------------------------------------------------------------
    def _recall_uri(self, recall_id):
        return self.namespace[f"recall/{self._encode(recall_id)}"]

    # Build the lookup key used to reuse vehicle resources
    # -----------------------------------------------------
    def _vehicle_key(self, row):
        return tuple(str(row.get(column)) for column in ("make", "model", "model_year"))

    # Add an RDF literal only when its source value is present
    # --------------------------------------------------------
    def _add_literal(self, graph, subject, predicate, value, datatype=None):
        if self._has_value(value):
            graph.add((subject, self.namespace[predicate], Literal(value, datatype=datatype)))

    # Return the first available value from a list of row columns
    # ------------------------------------------------------------
    def _first_value(self, row, *columns):
        for column in columns:
            value = row.get(column)
            if self._has_value(value):
                return value
        return None

    # Check whether a scalar value is suitable for RDF output
    # --------------------------------------------------------
    def _has_value(self, value):
        if value is None:
            return False
        result = pd.notna(value)
        return bool(result) if not hasattr(result, "__len__") else False

    # Encode a source value so it can safely form part of an RDF URI
    # --------------------------------------------------------------
    def _encode(self, value):
        return quote(str(value), safe="")
