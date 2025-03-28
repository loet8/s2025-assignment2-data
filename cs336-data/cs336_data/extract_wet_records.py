from warcio.archiveiterator import ArchiveIterator
import os

output_dir = "wet_records"
os.makedirs(output_dir, exist_ok=True)

with open("CC-MAIN-20180420081400-20180420101400-00118.warc.wet.gz", "rb") as f:
    count = 0
    for record in ArchiveIterator(f):
        if record.rec_type == "conversion":
            count += 1
            if count < 26:
                continue  
            if count > 50:
                break     
            filename = os.path.join(output_dir, f"record{count}.txt")
            content = record.content_stream().read().decode('utf-8', errors='ignore')
            with open(filename, "w", encoding="utf-8") as out:
                out.write(content)

print("Extracted records 26 through 50 into the 'wet_records' directory.")
