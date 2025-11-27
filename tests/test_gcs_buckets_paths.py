from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/able-bazaar-477311-s0-e7b9ea511911.json"

BUCKET_NAME = "mlops-tiker-bucket"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

print("=== All objects in bucket (first 200) ===")
for i, blob in enumerate(bucket.list_blobs()):
    print(blob.name)
    if i >= 199:
        print("... truncated ...")
        break

print("\n=== Objects that look like processed weekly CSVs ===")
for blob in bucket.list_blobs():
    name = blob.name.lower()
    if name.endswith("_weekly_clean.csv"):
        print(blob.name)

print("\n=== Suggested prefixes for GCS_PROCESSED_PREFIX ===")
prefixes = set()
for blob in bucket.list_blobs():
    name = blob.name
    if name.endswith("_weekly_clean.csv"):
        # take everything before the filename
        parts = name.split("/")
        if len(parts) > 1:
            prefix = "/".join(parts[:-1])
            prefixes.add(prefix)

for p in prefixes:
    print(f"gs://{BUCKET_NAME}/{p}")
