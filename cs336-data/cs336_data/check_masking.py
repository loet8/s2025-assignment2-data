import os
import pandas as pd
from tests import adapters

wet_records_folder = "wet_records"

file_names = os.listdir(wet_records_folder)
wet_files = [os.path.join(wet_records_folder, f) for f in file_names if f.endswith(".txt")]

results = []
for file_path in wet_files:
    print(f"Processing file: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().strip()

    original_text = text[:300]

    masked_email_text, email_count = adapters.run_mask_emails(text)
    masked_phone_text, phone_count = adapters.run_mask_phone_numbers(masked_email_text)
    masked_ip_text, ip_count = adapters.run_mask_ips(masked_phone_text)

    total_replacements = email_count + phone_count + ip_count

    if total_replacements > 0:
        results.append({
            "File": file_path,
            "Original Text": original_text,
            "Masked Text": masked_ip_text[:300],
            "Email Count": email_count,
            "Phone Count": phone_count,
            "IP Count": ip_count,
            "Total Replacements": total_replacements
        })

# Display Results
df_results = pd.DataFrame(results)

if df_results.empty:
    print("No records contained PII data that matched the patterns.")
else:
    df_sample = df_results.sample(min(20, len(df_results)), random_state=42)
    output_file = "pii_masking_analysis.csv"
    df_sample.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
