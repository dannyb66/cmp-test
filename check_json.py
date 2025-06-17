import zipfile
import json
import sys
import os

def get_json_keys_from_zip(zip_path):
    key_map = {}

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            # Skip directories
            if file_info.is_dir():
                continue

            if file_info.filename.lower().endswith('.json'):
                try:
                    with zip_ref.open(file_info) as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            key_map[file_info.filename] = list(data.keys())
                        else:
                            key_map[file_info.filename] = ['<not a dict>']
                except Exception as e:
                    key_map[file_info.filename] = [f'<error: {str(e)}>']

    return key_map

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <zip_file_path>")
        sys.exit(1)

    zip_file = sys.argv[1]

    if not os.path.isfile(zip_file):
        print(f"Error: File not found - {zip_file}")
        sys.exit(1)

    keys = get_json_keys_from_zip(zip_file)
    for filename, json_keys in keys.items():
        print(f"{filename}: {json_keys}")
