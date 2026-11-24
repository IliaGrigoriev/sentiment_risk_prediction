import base64
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class TriplestoreClient:
    """Upload RDF and run SPARQL queries over standard HTTP endpoints."""

    # Configure the graph-store, query, and optional authentication endpoints
    # ------------------------------------------------------------------------
    def __init__(self, graph_store_url, query_url=None, username=None, password=None):
        self.graph_store_url = graph_store_url
        self.query_url = query_url or graph_store_url
        self.username = username
        self.password = password

    # Replace the selected graph-store graph with the supplied RDF graph
    # -----------------------------------------------------------------
    def upload(self, graph, graph_uri=None, rdf_format="turtle"):
        url = self._with_graph_uri(self.graph_store_url, graph_uri)
        serialized = graph.serialize(format=rdf_format)
        body = serialized if isinstance(serialized, bytes) else serialized.encode("utf-8")
        request = Request(
            url,
            data=body,
            method="PUT",
            headers={"Content-Type": "text/turtle; charset=utf-8"},
        )
        self._add_auth(request)
        with urlopen(request) as response:
            return response.status

    # Execute a SPARQL query and return its decoded JSON result
    # ---------------------------------------------------------
    def query(self, sparql):
        body = urlencode({"query": sparql}).encode("utf-8")
        request = Request(
            self.query_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/sparql-results+json",
            },
        )
        self._add_auth(request)
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))

    # Add an optional named-graph parameter to a graph-store URL
    # ----------------------------------------------------------
    def _with_graph_uri(self, url, graph_uri):
        if not graph_uri:
            return url
        separator = "&" if "?" in url else "?"
        parameter = urlencode({"graph": graph_uri})
        return f"{url}{separator}{parameter}"

    # Attach an HTTP Basic Authentication header when configured
    # -----------------------------------------------------------
    def _add_auth(self, request):
        if self.username is None:
            return
        credentials = f"{self.username}:{self.password or ''}".encode("utf-8")
        token = base64.b64encode(credentials).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
