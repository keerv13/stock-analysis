
import os
import uuid

import pandas as pd
from dotenv import load_dotenv

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
)

load_dotenv()

MODEL = os.environ["EMBEDDINGS_MODEL"]
INDEX_NAME = os.getenv(
    "AISearch_INDEX_NAME", "stockpilot-index"
)

# Authenticate to your Azure AI project.
credential = DefaultAzureCredential()

project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"],
    credential=credential,
)

# Embedding service from the Azure AI project.
embeddings = project.inference.get_embeddings_client()

# Obtain the connected Azure AI Search service.
connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH,
    include_credentials=True,
)

search_endpoint = connection.endpoint_url
search_credential = AzureKeyCredential(connection.key)

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=search_credential,
)

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=INDEX_NAME,
    credential=search_credential,
)


def embed_text(text: str) -> list[float]:
    """Generate a vector using the configured embedding model."""
    result = embeddings.embed(
        input=text,
        model=MODEL,
    )
    return result.data[0].embedding


def create_index_definition(
    index_name: str,
    dimensions: int,
) -> SearchIndex:
    """Define the fields and vector-search configuration."""

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
        ),
        SearchableField(
            name="Company",
            type=SearchFieldDataType.String,
        ),
        SearchableField(
            name="Content",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="Date",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="Ticker",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="High",
            type=SearchFieldDataType.Double,
        ),
        SimpleField(
            name="Low",
            type=SearchFieldDataType.Double,
        ),
        SimpleField(
            name="Dividends",
            type=SearchFieldDataType.Double,
        ),
        SimpleField(
            name="Stock_Splits",
            type=SearchFieldDataType.Double,
        ),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="stockVectorProfile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="stockHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters={
                    "metric": VectorSearchAlgorithmMetric.COSINE,
                },
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="stockVectorProfile",
                algorithm_configuration_name="stockHnsw",
            )
        ],
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )


def create_docs_from_csv(path: str) -> list[dict]:
    """Turn each CSV row into a searchable stock record."""

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "Company" not in df.columns:
        raise ValueError("CSV must contain a Company column.")

    documents = []

    for row in df.to_dict(orient="records"):
        company = str(row.get("Company", "Unknown"))
        ticker = str(row.get("Ticker", "") or "").upper()
        date = str(row.get("Date", ""))

        def number(field: str) -> float:
            value = pd.to_numeric(
                row.get(field), errors="coerce"
            )
            return 0.0 if pd.isna(value) else float(value)

        high = number("High")
        low = number("Low")
        dividends = number("Dividends")
        splits = number("Stock Splits")

        # Include all important information in the embedded text.
        content = (
            f"Company: {company}. "
            f"Ticker: {ticker or 'Unavailable'}. "
            f"Date: {date}. "
            f"High: {high}. Low: {low}. "
            f"Dividends: {dividends}. "
            f"Stock splits: {splits}."
        )

        documents.append({
            "id": str(uuid.uuid4()),
            "Company": company,
            "Content": content,
            "Date": date,
            "Ticker": ticker,
            "High": high,
            "Low": low,
            "Dividends": dividends,
            "Stock_Splits": splits,
            "contentVector": embed_text(content),
        })

    return documents


def create_index_from_csv(
    index_name: str,
    csv_file: str,
) -> None:
    """Create an index and upload stock records."""

    documents = create_docs_from_csv(csv_file)

    if not documents:
        raise ValueError("The CSV contains no stock records.")

    # Infer the actual dimension from the configured embedding model.
    dimensions = len(documents[0]["contentVector"])

    try:
        index_client.get_index(index_name)
        print(f"Index '{index_name}' already exists.")
        print("Keeping it unchanged to avoid accidental data loss.")
        return
    except Exception:
        # In production, catch ResourceNotFoundError specifically.
        pass

    definition = create_index_definition(
        index_name=index_name,
        dimensions=dimensions,
    )

    index_client.create_index(definition)

    client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=search_credential,
    )

    # Upload in batches instead of sending one enormous request.
    batch_size = 100

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        results = client.upload_documents(documents=batch)

        failures = [
            item.key for item in results if not item.succeeded
        ]

        if failures:
            raise RuntimeError(
                f"Failed to upload {len(failures)} documents."
            )

    print(
        f"Uploaded {len(documents)} documents "
        f"to index '{index_name}'."
    )


def search_stock_records(
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve stock records using vector similarity."""

    query_vector = embed_text(question)

    from azure.search.documents.models import VectorizedQuery

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="contentVector",
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=[
            "Company",
            "Content",
            "Date",
            "Ticker",
            "High",
            "Low",
            "Dividends",
            "Stock_Splits",
        ],
        top=top_k,
    )

    return [
        {
            "company": item.get("Company"),
            "content": item.get("Content"),
            "date": item.get("Date"),
            "ticker": item.get("Ticker"),
            "high": item.get("High"),
            "low": item.get("Low"),
            "dividends": item.get("Dividends"),
            "stock_splits": item.get("Stock_Splits"),
            "score": item.get("@search.score"),
        }
        for item in results
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-name",
        default=INDEX_NAME,
    )
    parser.add_argument(
        "--csv-file",
        default="assets/stock_details_5_years.csv",
    )

    args = parser.parse_args()

    create_index_from_csv(
        args.index_name,
        args.csv_file,
    )