# StockPilot AI

StockPilot AI is a portfolio project for exploring three Bursa Malaysia stocks—Maybank, MR D.I.Y., and Tenaga Nasional—and asking questions about their FY2025 annual reports.

## What it does

- Loads historical stock-price CSVs and displays price charts, moving averages, approximate returns, and cross-stock performance comparisons in Streamlit.
- Ingests annual-report PDFs, splits their text into chunks, creates embeddings with Azure OpenAI `text-embedding-3-small`, and stores the vectors in Azure AI Search.
- Uses LangChain and an Azure-hosted chat-model deployment to answer questions using retrieved report passages, with source filenames and PDF page references.

The dashboard uses the CSVs you provide; it is not a live market-data feed. AI answers should be checked against the cited report passages and are not investment advice.

## Setup

1. Create and activate a virtual environment.

   **Windows PowerShell:**

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in the values from your Azure resources.

   ```powershell
   Copy-Item .env.example .env
   ```

3. Confirm that the Azure OpenAI chat and embedding deployments exist, and that your Azure AI Search service is active. The deployment names in `.env` must match the names in your Azure portal.

4. Ingest the annual reports into Azure AI Search:

   ```powershell
   python ingest_reports.py
   ```

5. Optionally test retrieval:

   ```powershell
   python test_report_search.py
   ```

6. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

## Environment variables

See `.env.example` for the template. The key settings are:

- `AZURE_OPENAI_ENDPOINT`: base endpoint of the Azure OpenAI resource; use the resource endpoint, not a full `/openai/v1/responses` URL.
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key.
- `AZURE_OPENAI_DEPLOYMENT`: exact chat-model deployment name.
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: exact embedding deployment name, e.g. `text-embedding-3-small`.
- `AZURE_SEARCH_ENDPOINT`: Azure AI Search service endpoint, usually `https://<service-name>.search.windows.net`.
- `AZURE_SEARCH_ADMIN_KEY`: Azure AI Search admin key for local index creation and ingestion.
- `AZURE_SEARCH_INDEX_NAME`: index name used by the scripts, e.g. `stockpilot-reports`.

The current `/openai/v1/` client setup does not use `AZURE_OPENAI_API_VERSION`; it is included as a reference for code paths that still use the versioned Azure OpenAI API. Do not add real secrets to Git or share your `.env` file. Keep `.env` in `.gitignore` and use app settings or managed identity for deployed environments where possible.

## Notes

- The report index must be created in an Azure AI Search service. An Azure OpenAI endpoint is not an Azure AI Search endpoint.
- The embedding dimensions configured during ingestion must match the dimensions used for query embeddings. This project uses 512 dimensions if your scripts are configured as in the setup discussed during development.
- If you change chunking, embedding dimensions, index schema, or company metadata, recreate or deliberately migrate the report index before re-ingesting to avoid duplicate or incompatible records.
