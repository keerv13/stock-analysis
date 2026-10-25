
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()

# Azure OpenAI endpoint: use your resource's base endpoint,
# not the /openai/v1/responses URL.
OPENAI_BASE_URL = (
    os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    + "/openai/v1/"
)

# Embeddings must match those used to index the reports.
embeddings = OpenAIEmbeddings(
    model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    base_url=OPENAI_BASE_URL,
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    dimensions=512,
)

# Chat model deployment.
llm = ChatOpenAI(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    base_url=OPENAI_BASE_URL,
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.1,
)

# Azure AI Search.
search_client = SearchClient(
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

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are StockPilot AI, an assistant that analyzes
Malaysian listed companies using their annual reports.

Answer using only the supplied report passages.
Do not invent financial figures, company statements, or citations.
If the passages do not contain enough evidence, say so clearly.

Cite supporting passages using their numbered references,
for example [1] or [2].
Distinguish reported facts from your own explanations.
You provide information for research, not personalized
investment advice.

Report passages:
{context}"""
    ),
    ("human", "{question}"),
])


def search_passages(question: str, top_k: int = 5):
    """Retrieve relevant annual-report passages."""
    query_vector = embeddings.embed_query(question)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="contentVector",
    )

    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=["company", "source", "page", "content"],
        top=top_k,
    )

    return list(results)


def answer_question(question: str):
    """Retrieve evidence, ask the chat model, and return citations."""
    passages = search_passages(question)

    if not passages:
        return (
            "I couldn't find relevant passages in the indexed reports.",
            [],
        )

    context_parts = []
    sources = []

    for number, passage in enumerate(passages, start=1):
        company = passage.get("company", "Unknown company")
        source = passage.get("source", "Unknown source")
        page = passage.get("page", "?")
        content = passage.get("content", "")

        context_parts.append(
            f"[{number}] Company: {company}\n"
            f"Source: {source}\n"
            f"PDF page: {page}\n"
            f"Passage:\n{content}"
        )

        sources.append({
            "reference": number,
            "company": company,
            "source": source,
            "page": page,
            "score": passage.get("@search.score"),
        })

    context = "\n\n".join(context_parts)

    messages = PROMPT.format_messages(
        question=question,
        context=context,
    )

    response = llm.invoke(messages)

    return response.content, sources


if __name__ == "__main__":
    question = input("Ask StockPilot AI: ").strip()

    if not question:
        print("Please enter a question.")
    else:
        try:
            answer, sources = answer_question(question)

            print("\n=== StockPilot AI Answer ===\n")
            print(answer)

            print("\n=== Retrieved Sources ===")
            for source in sources:
                print(
                    f"[{source['reference']}] "
                    f"{source['company']} | "
                    f"{source['source']} | "
                    f"PDF page {source['page']}"
                )

        except Exception as exc:
            print(f"Request failed: {exc}")
            raise