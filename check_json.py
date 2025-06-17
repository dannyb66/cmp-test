import zipfile
import json
import sys
import os
import io

def get_json_keys_from_zip(zip_path):
    key_map = {}

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                continue

            filename = file_info.filename
            if not filename.lower().endswith('.json'):
                continue

            try:
                # Use buffered reader for large files
                with zip_ref.open(file_info) as raw:
                    buffered_reader = io.TextIOWrapper(raw, encoding='utf-8', errors='ignore')
                    # Use JSON streaming approach, expecting top-level object only
                    data = json.load(buffered_reader)
                    if isinstance(data, dict):
                        key_map[filename] = list(data.keys())
                    else:
                        key_map[filename] = ['<not a dict>']
            except Exception as e:
                key_map[filename] = [f'<error: {str(e)}>']

    return key_map

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <zip_file_path>", file=sys.stderr)
        sys.exit(1)

    zip_file = sys.argv[1]

    if not os.path.isfile(zip_file):
        print(f"Error: File not found - {zip_file}", file=sys.stderr)
        sys.exit(1)

    result = get_json_keys_from_zip(zip_file)

    # Output as pretty JSON (good for both humans and machines)
    print(json.dumps(result, indent=2))
