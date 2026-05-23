"""
Azure Blob Storage Integration
================================
Uploads results (predictions CSV, figures, model artifacts) to Azure Blob Storage.
Uses Azure Identity (DefaultAzureCredential) — no hardcoded secrets.

Cost note: Azure Blob Storage LRS (locally redundant) is the cheapest tier.
At ~$0.002/GB/month, this experiment's data will cost <$0.01/month total.

Setup:
    1. Create a Storage Account in Azure Portal (LRS, Standard, General Purpose v2)
    2. Copy the account name to .env:  AZURE_STORAGE_ACCOUNT=<name>
    3. Run:  az login  (or set AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID)
    4. Assign yourself "Storage Blob Data Contributor" on the storage account
    5. python azure/azure_storage.py
"""

import os
import glob
from dotenv import load_dotenv

load_dotenv()


def upload_results(
    storage_account: str | None = None,
    container_name:  str        = "pipeline-anomaly-results",
    local_dir:       str        = "results",
) -> list[str]:
    """
    Upload all files in `local_dir` to Azure Blob Storage.
    Returns list of uploaded blob URLs.
    """
    try:
        from azure.storage.blob import BlobServiceClient
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("[azure_storage] azure-storage-blob not installed. "
              "Run: pip install azure-storage-blob azure-identity")
        return []

    account = storage_account or os.getenv("AZURE_STORAGE_ACCOUNT")
    if not account:
        print("[azure_storage] No storage account specified. "
              "Set AZURE_STORAGE_ACCOUNT in .env or pass as argument.")
        return []

    url        = f"https://{account}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    client     = BlobServiceClient(url, credential=credential)

    # Create container if missing
    try:
        client.create_container(container_name)
        print(f"[azure_storage] Created container: {container_name}")
    except Exception:
        pass   # already exists

    container_client = client.get_container_client(container_name)
    uploaded         = []

    for local_path in glob.glob(os.path.join(local_dir, "**", "*"), recursive=True):
        if os.path.isfile(local_path):
            blob_name = local_path.replace("\\", "/")
            with open(local_path, "rb") as f:
                container_client.upload_blob(blob_name, f, overwrite=True)
            url_out = f"{url}/{container_name}/{blob_name}"
            uploaded.append(url_out)
            print(f"[azure_storage] Uploaded: {blob_name}")

    print(f"\n[azure_storage] Done. {len(uploaded)} files uploaded.")
    return uploaded


def download_results(
    storage_account: str | None = None,
    container_name:  str        = "pipeline-anomaly-results",
    local_dir:       str        = "results_from_azure",
) -> None:
    """Download all blobs from container to local directory."""
    try:
        from azure.storage.blob import BlobServiceClient
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("[azure_storage] azure-storage-blob not installed.")
        return

    account    = storage_account or os.getenv("AZURE_STORAGE_ACCOUNT")
    url        = f"https://{account}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    client     = BlobServiceClient(url, credential=credential)
    container_client = client.get_container_client(container_name)

    os.makedirs(local_dir, exist_ok=True)
    for blob in container_client.list_blobs():
        local_path = os.path.join(local_dir, blob.name.replace("/", os.sep))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(container_client.download_blob(blob.name).readall())
        print(f"[azure_storage] Downloaded: {blob.name}")


if __name__ == "__main__":
    upload_results()
