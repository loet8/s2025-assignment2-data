import os
from tests import adapters

wet_records_folder = "wet_records"
file_names = os.listdir(wet_records_folder)
wet_files = [os.path.join(wet_records_folder, f) for f in file_names if f.endswith(".txt")]

results = []  
for file_path in wet_files:
    print(f"Processing file: {file_path}")
    with open(file_path, 'rb') as f:
        html_bytes = f.read()

    text = adapters.run_extract_text_from_html_bytes(html_bytes)

    quality_passed = adapters.run_gopher_quality_filter(text)

    language, confidence = adapters.run_identify_language(text)
    
    results.append((file_path, text, quality_passed, language, confidence))
    print(f"File: {file_path}\n  Quality Filter: {quality_passed}\n  Detected Language: {language}\n  Confidence: {confidence:.2f}\n")
