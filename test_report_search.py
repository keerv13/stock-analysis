

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()

# Embedding model — must match the one used during ingestion
embeddings = OpenAIEmbeddings(
    model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    base_url=(
        os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        + "/openai/v1/"
    ),
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    dimensions=512,
)

# Azure AI Search client
client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=os.getenv(
        "AZURE_SEARCH_INDEX_NAME",
        "stockpilot-reports",
    ),
    credential=AzureKeyCredential(
        os.environ["AZURE_SEARCH_ADMIN_KEY"]
    ),
    api_version="2024-07-01",
)

question = (
    "What were the main business risks discussed "
    "in Maybank's 2025 annual report?"
)

# Convert the question to a vector
query_vector = embeddings.embed_query(question)

vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=5,
    fields="contentVector",
)

# Hybrid search combines keyword and vector retrieval
results = client.search(
    search_text=question,
    vector_queries=[vector_query],
    select=["company", "source", "page", "content"],
    top=5,
)

matches = list(results)

if not matches:
    print("No matching report passages found.")
else:
    for i, result in enumerate(matches, start=1):
        print(f"\n--- Result {i} ---")
        print("Company:", result["company"])
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Score:", result.get("@search.score"))
        print("Passage:", result["content"][:700])