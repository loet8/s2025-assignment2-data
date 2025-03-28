from tests import adapters
from warcio.archiveiterator import ArchiveIterator
import gzip

warc_file_path = "/Users/tiffanyloe/Desktop/ECE 491B/s2025-assignment2-data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz"  

with gzip.open(warc_file_path, "rb") as f:
    for record in ArchiveIterator(f):
        if record.rec_type == "response":
            html_bytes = record.content_stream().read()
            extracted_text = adapters.run_extract_text_from_html_bytes(html_bytes)
            print("Extracted text from the record:")
            print(extracted_text)
            break

