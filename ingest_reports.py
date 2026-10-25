
import os
import uuid
from pathlib import Path

import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchableField,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

load_dotenv()

REPORTS_DIR = Path("assets/reports")
INDEX_NAME = os.getenv(
    "AZURE_SEARCH_INDEX_NAME", "stockpilot-reports"
)

# Initialize Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    dimensions=512,
    api_version=os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-10-21"
    ),
    azure_deployment=os.environ[
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ],
)

search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]
credential = AzureKeyCredential(search_key)

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=credential,
    api_version="2024-07-01",
)

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=INDEX_NAME,
    credential=credential,
    api_version="2024-07-01",
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=200,
)


def load_pdf_chunks(pdf_path: Path):
    """Extract text page by page and preserve page numbers."""
    chunks = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            for chunk in splitter.split_text(text):
                chunks.append({
                    "company": pdf_path.stem.split("_2025")[0],
                    "source": pdf_path.name,
                    "page": page_number,
                    "content": chunk,
                })

    return chunks


def create_index(vector_dimensions: int):
    """Create the index only if it does not already exist."""
    existing = list(index_client.list_index_names())

    if INDEX_NAME in existing:
        print(f"Index '{INDEX_NAME}' already exists.")
        return

    index = SearchIndex(
        name=INDEX_NAME,
        fields=[
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
            ),
            SearchableField(
                name="company",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchableField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="page",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
            ),
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single
                ),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="stockpilot-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="stockpilot-hnsw"
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="stockpilot-profile",
                    algorithm_configuration_name="stockpilot-hnsw",
                )
            ],
        ),
    )

    index_client.create_index(index)
    print(f"Created index: {INDEX_NAME}")


def main():
    if not REPORTS_DIR.exists():
        raise FileNotFoundError(
            f"Reports folder not found: {REPORTS_DIR}"
        )

    pdf_files = sorted(REPORTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {REPORTS_DIR}"
        )

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name}...")
        chunks = load_pdf_chunks(pdf_path)
        print(f"Extracted {len(chunks)} chunks.")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(
            "No text extracted. Some PDFs may be scanned images "
            "and require OCR."
        )

    # Determine dimensions from the deployed embedding model.
    test_vector = embeddings.embed_query("StockPilot AI test")
    create_index(len(test_vector))

    # Embed and upload in batches to control memory and request size.
    batch_size = 32

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]
        texts = [item["content"] for item in batch]
        vectors = embeddings.embed_documents(texts)

        documents = []

        for item, vector in zip(batch, vectors):
            documents.append({
                "id": str(uuid.uuid4()),
                "company": item["company"],
                "source": item["source"],
                "page": item["page"],
                "content": item["content"],
                "contentVector": vector,
            })

        results = search_client.upload_documents(
            documents=documents
        )

        failed = [
            result.key
            for result in results
            if not result.succeeded
        ]

        if failed:
            print(f"Failed document IDs: {failed}")

        print(
            f"Processed {min(start + batch_size, len(all_chunks))}"
            f"/{len(all_chunks)} chunks"
        )

    print("Report ingestion complete.")


if __name__ == "__main__":
    main()