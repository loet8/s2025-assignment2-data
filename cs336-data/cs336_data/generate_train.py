#!/usr/bin/env python3
import os
import asyncio
import aiohttp
import logging
from warcio.archiveiterator import ArchiveIterator
from tests import adapters

logging.basicConfig(level=logging.INFO)
MAX_EXAMPLES = 1000  

async def fetch(session, url, semaphore):
    async with semaphore:
        try:
            undesired_exts = [".pdf", ".xlsx", ".doc", ".docx", ".ppt", ".pptx", ".zip", ".rar"]
            if any(url.lower().endswith(ext) for ext in undesired_exts):
                logging.info(f"Skipping URL with undesired extension: {url}")
                return None
            async with session.get(url, timeout=5) as response:
                if response.status != 200:
                    logging.warning(f"HTTP error {response.status} for {url}")
                    return None
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    logging.warning(f"Skipping non-html content for {url}: {content_type}")
                    return None
                response.encoding = 'utf-8'
                return await response.read()
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

async def process_wiki_urls(url_file, label, output_file, max_examples=1000, url_limit=10000):
    with open(url_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    urls = urls[:url_limit]
    logging.info(f"Processing up to {url_limit} URLs to collect {max_examples} high-quality examples.")
    
    semaphore = asyncio.Semaphore(200)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)

    with open(output_file, "a", encoding="utf-8") as fout:
        count = 0
        rejected_count = 0
        for i, data in enumerate(results):
            if data is None:
                continue
            try:
                extracted = adapters.run_extract_text_from_html_bytes(data)
                if extracted and adapters.run_gopher_quality_filter(extracted):
                    single_line = " ".join(extracted.split())
                    fout.write(f"__label__{label} {single_line}\n")
                    count += 1
                    if count >= max_examples:
                        break  
                else:
                    rejected_count += 1
            except Exception as e:
                logging.error(f"Error processing URL {urls[i]}: {e}")
    
    logging.info(f"Processed {count} high-quality examples.")
    logging.info(f"Rejected {rejected_count} Wiki URLs as low quality.")



def process_wet_file(wet_path, label, output_file, language_threshold=0.5, max_examples=MAX_EXAMPLES):
    total_records = 0
    count = 0
    batch = []  
    batch_size = 100  

    with open(wet_path, "rb") as f:
        for record in ArchiveIterator(f):
            total_records += 1  
            if count >= max_examples:  
                break

            if record.rec_type != "conversion":
                continue

            try:
                text = record.content_stream().read().decode('utf-8', errors='ignore').strip()

                print(f"Record {total_records} - Length: {len(text)}")

                if not text:
                    continue
                
                lang, conf = adapters.run_identify_language(text)
                print(f"Record {total_records} - Language: {lang} | Confidence: {conf:.2f}")

                if lang != 'en' or conf < language_threshold:
                    continue

                if not adapters.run_gopher_quality_filter(text):
                    single_line = " ".join(text.split())
                    batch.append(f"__label__{label} {single_line}\n")

                    if len(batch) >= batch_size:
                        with open(output_file, "a", encoding="utf-8") as fout:
                            fout.writelines(batch)
                        batch.clear()  
                    
                    count += 1
            except Exception as e:
                logging.error(f"Error processing a WET record: {e}")

    if batch:
        with open(output_file, "a", encoding="utf-8") as fout:
            fout.writelines(batch)

    print(f"Total WET records scanned: {total_records}")
    logging.info(f"Processed {count} low-quality examples from the WET file.")

def main():
    wiki_url_file = "/Users/tiffanyloe/Desktop/ECE 491B/s2025-assignment2-data/enwiki-20240420-extracted_urls.txt"
    wet_path = '/Users/tiffanyloe/Desktop/ECE 491B/s2025-assignment2-data/CC-MAIN-20180420081400-20180420101400-00118.warc.wet.gz'
    training_file = "quality_train.txt"
    if os.path.exists(training_file):
        os.remove(training_file)

    asyncio.run(process_wiki_urls(wiki_url_file, "high", training_file, MAX_EXAMPLES))

    process_wet_file(wet_path, "low", training_file, language_threshold=0.5, max_examples=MAX_EXAMPLES)
    
    logging.info(f"Training file created at: {training_file}")

if __name__ == "__main__":
    main()
