import os
from google.cloud import storage

# Local folder with your processed CSVs
LOCAL_PROCESSED_DIR = "data/processed"

# GCS bucket + prefix where you want human-readable files
BUCKET_NAME = "mlops-tiker-bucket"
GCS_PREFIX = "processed"  # will create gs://mlops-tiker-bucket/processed/...

# Path to your service account JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/able-bazaar-477311-s0-e7b9ea511911.json"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

for fname in os.listdir(LOCAL_PROCESSED_DIR):
    if not fname.endswith("_weekly_clean.csv"):
        continue
    local_path = os.path.join(LOCAL_PROCESSED_DIR, fname)
    blob_path = f"{GCS_PREFIX}/{fname}"  # e.g. processed/BIC_weekly_clean.csv
    print(f"Uploading {local_path} -> gs://{BUCKET_NAME}/{blob_path}")
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

print("Done.")
