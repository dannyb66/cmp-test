# Run this cell in Google Colab

import zipfile
import json
import io
import os
import tempfile
import urllib.request
from urllib.parse import urlparse
from google.colab import files

def download_zip(url):
    """
    Downloads a ZIP file from a URL to a temporary file.
    Returns the path to the temporary file.
    """
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download file. HTTP status: {response.status}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                tmp_file.write(response.read())
                return tmp_file.name
    except Exception as e:
        print(f"Download error: {e}")
        return None

def get_json_keys_from_zip(zip_path):
    """
    Extracts top-level keys from JSON files inside a ZIP archive.
    """
    key_map = {}

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                continue

            filename = file_info.filename
            if not filename.lower().endswith('.json'):
                continue

            try:
                with zip_ref.open(file_info) as raw:
                    buffered_reader = io.TextIOWrapper(raw, encoding='utf-8', errors='ignore')
                    data = json.load(buffered_reader)

                    if isinstance(data, dict):
                        key_map[filename] = list(data.keys())
                    else:
                        key_map[filename] = ['<not a dict>']
            except Exception as e:
                key_map[filename] = [f'<error: {str(e)}>']

    return key_map

# --- Prompt and Run ---
zip_input = input("Enter ZIP file URL or local filename (or upload if not already): ").strip()

if zip_input.startswith("http://") or zip_input.startswith("https://"):
    zip_path = download_zip(zip_input)
else:
    if not os.path.isfile(zip_input):
        print("File not found locally. Please upload it now.")
        uploaded = files.upload()
        zip_input = list(uploaded.keys())[0]
    zip_path = zip_input

if zip_path and os.path.isfile(zip_path):
    result = get_json_keys_from_zip(zip_path)
    print(json.dumps(result, indent=2))
else:
    print("Failed to process the ZIP file.")
