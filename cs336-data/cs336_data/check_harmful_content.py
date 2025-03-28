import os
import pandas as pd
from tests import adapters  


input_dir = '/Users/tiffanyloe/Desktop/ECE 491B/s2025-assignment2-data/wet_records'

classified_results = []

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            extracted_text = file.read().strip()

            if not extracted_text:
                continue

            nsfw_label, nsfw_score = adapters.run_classify_nsfw(extracted_text)
            toxic_label, toxic_score = adapters.run_classify_toxic_speech(extracted_text)

            classified_results.append({
                "Filename": filename,
                "Extracted Text": extracted_text[:300],  
                "NSFW Label": nsfw_label,
                "NSFW Confidence": round(nsfw_score, 2),
                "toxic Speech Label": toxic_label,
                "toxic Speech Confidence": round(toxic_score, 2)
            })

df_results = pd.DataFrame(classified_results)

print(df_results)

output_file = "harmful_content_analysis.csv"
df_results.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")