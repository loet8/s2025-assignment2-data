import os
from tests import adapters

wet_records_folder = "wet_records"

file_names = os.listdir(wet_records_folder)
wet_files = [os.path.join(wet_records_folder, f) for f in file_names if f.endswith(".txt")]

for file_path in wet_files:
    print(f"Processing file: {file_path}")

results = []
for file_path in wet_files:

    with open(file_path, 'rb') as f:
        html_bytes = f.read()

    text = adapters.run_extract_text_from_html_bytes(html_bytes)

    language, confidence = adapters.run_identify_language(text)
    results.append((file_path, language, confidence))
    print(f"File: {file_path}\n  Detected Language: {language}\n  Average Confidence: {confidence:.2f}\n")

english_docs = [r for r in results if r[1] == 'en']
fraction_english = len(english_docs) / len(results) if results else 0
print(f"Fraction of documents that are English: {fraction_english:.2f}")