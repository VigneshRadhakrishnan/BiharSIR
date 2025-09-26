# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "dask[distributed]",
#   "pandas",
#   "google-cloud-vision",
#   "pdf2image"
# ]
# ///
import os
import glob
import io
import re
import logging
from typing import List, Dict, Optional, Tuple
from timeit import default_timer as timer
from datetime import timedelta
import pandas as pd

# --- New: Dask for parallelism ---
import dask.bag as db
from dask.distributed import Client, LocalCluster


logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(processName)s[%(process)d] - %(message)s")


class PDFTableExtractor:
    """
    Same extractor as your original, unchanged logic.
    Only plugged into a Dask-powered main() below.
    """

    def __init__(self, google_credentials_path: str):
        self.vision_client = self._setup_google_vision(google_credentials_path)
        self.table_template = None

    def _setup_google_vision(self, credentials_path: str):
        try:
            from google.cloud import vision
            if not os.path.exists(credentials_path):
                logging.error(f"Google credentials not found at: {credentials_path}")
                return None
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            client = vision.ImageAnnotatorClient()
            logging.info("✅ Google Vision API initialized successfully.")
            return client
        except ImportError:
            logging.error("❌ google-cloud-vision is not installed. Please install: pip install google-cloud-vision")
            return None
        except Exception as e:
            logging.error(f"❌ Google Vision API setup failed: {e}")
            return None

    def _get_all_words_from_annotation(self, annotation) -> List[Tuple[str, int, object]]:
        words = []
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join([s.text for s in word.symbols])
                        avg_y = sum(v.y for v in word.bounding_box.vertices) // 4
                        words.append((word_text, avg_y, word.bounding_box))
        return words

    def _create_table_template(self, annotation) -> Optional[Dict[str, Tuple[int, int]]]:
        logging.info("🧠 Analyzing first PDF to create a reusable table layout template...")
        words = self._get_all_words_from_annotation(annotation)

        header_config = {
            "serial_no": ("क्रम",), "epic_no": ("ईपिक",), "name": ("मतदाता",),
            "relation_type": ("संबंध",), "relative_name": ("संबंधी",),
            "age": ("आयु",), "gender": ("लिंग",), "reason": ("अप्राप्ति",)
        }

        header_starts = {}
        for key, texts in header_config.items():
            found_words = [w for w in words if w[0] in texts]
            if not found_words:
                logging.warning(f"Template creation: Could not find header for '{key}'.")
                continue
            min_x = min(v.x for w in found_words for v in w[2].vertices)
            header_starts[key] = min_x

        if "epic_no" not in header_starts or "age" not in header_starts:
            logging.error("Failed to find essential headers (epic_no, age). Cannot create template.")
            return None

        sorted_headers = sorted(header_starts.items(), key=lambda item: item[1])

        template = {}
        for i, (key, start_x) in enumerate(sorted_headers):
            end_x = sorted_headers[i + 1][1] if i + 1 < len(sorted_headers) else 9999
            template[key] = (start_x, end_x)

        logging.info(f"✅ Reusable template created successfully: {template}")
        return template

    def _parse_page(self, annotation, template: Dict[str, Tuple[int, int]], filename: str) -> List[Dict]:
        words = self._get_all_words_from_annotation(annotation)
        if not words:
            return []

        rows = []
        current_row = []
        sorted_words = sorted(words, key=lambda w: (w[1], w[2].vertices[0].x))
        if not sorted_words:
            return []

        last_y = sorted_words[0][1]
        for text, y, box in sorted_words:
            if abs(y - last_y) > 20 and current_row:
                rows.append(sorted(current_row, key=lambda w: w[2].vertices[0].x))
                current_row = []
            current_row.append((text, y, box))
            last_y = y
        if current_row:
            rows.append(sorted(current_row, key=lambda w: w[2].vertices[0].x))

        extracted_data = []
        for row in rows:
            first_word = row[0][0] if row else ''
            if not first_word.isdigit():
                continue

            row_data = {key: [] for key in template.keys()}
            for text, _, box in row:
                word_center_x = (box.vertices[0].x + box.vertices[1].x) / 2
                for col, (start_x, end_x) in template.items():
                    if start_x <= word_center_x < end_x:
                        row_data[col].append(text)

            epic_no_raw = "".join(row_data.get('epic_no', []))
            epic_no_clean = re.sub(r'[^A-Z0-9/]', '', epic_no_raw)

            age_raw = "".join(row_data.get('age', []))
            age_digits = ''.join(filter(str.isdigit, age_raw))
            if len(age_digits) > 2:
                if age_digits.startswith('10') or age_digits.startswith('11'):
                    age_digits = age_digits[:3]
                else:
                    age_digits = age_digits[:2]
            try:
                if age_digits and not (18 <= int(age_digits) <= 119):
                    age_digits = ""
            except ValueError:
                age_digits = ""

            age_text_for_gender = "".join(row_data.get('age', []))
            gender_text_for_gender = "".join(row_data.get('gender', []))
            combined_search_text = age_text_for_gender + " " + gender_text_for_gender
            combined_search_text = combined_search_text.replace('>', 'M')
            gender_match = re.search(r'(M|F)', combined_search_text)
            gender_clean = gender_match.group(1) if gender_match else ""

            reason_clean = " ".join(row_data.get('reason', []))

            if len(epic_no_clean) > 6:
                extracted_data.append({
                    'filename': filename,
                    'epic_no': epic_no_clean,
                    'age': age_digits,
                    'gender': gender_clean,
                    'reason': reason_clean
                })
        return extracted_data

    def extract_data_from_pdf(self, pdf_path: str, create_template: bool = False, create_debug_file: bool = False) -> List[Dict]:
        if not self.vision_client:
            return []

        try:
            from pdf2image import convert_from_path
            from google.cloud import vision

            logging.info(f"🔄 Converting {os.path.basename(pdf_path)} to images...")
            pages = convert_from_path(pdf_path, dpi=300)

            all_data = []
            for page_num, page_image in enumerate(pages, 1):
                logging.info(f"  🔍 Processing page {page_num}/{len(pages)}...")
                img_byte_arr = io.BytesIO()
                page_image.save(img_byte_arr, format='PNG')
                image = vision.Image(content=img_byte_arr.getvalue())
                response = self.vision_client.document_text_detection(image=image)
                if response.error.message:
                    logging.error(f"  ❌ Vision API error on page {page_num}: {response.error.message}")
                    continue

                annotation = response.full_text_annotation
                if not annotation:
                    logging.warning(f"  ⚠️ Page {page_num}: No text annotations returned.")
                    continue

                if create_template and page_num == 1:
                    if create_debug_file:
                        debug_file = "debug_page_1_raw_text.txt"
                        with open(debug_file, "w", encoding="utf-8") as f:
                            f.write(annotation.text)
                        logging.info(f"💾 Saved raw text for the first page of the first PDF to '{debug_file}'")

                    self.table_template = self._create_table_template(annotation)
                    if not self.table_template:
                        logging.error("❌ Failed to create table template. Aborting.")
                        return []

                page_data = self._parse_page(annotation, self.table_template, os.path.basename(pdf_path))
                if page_data:
                    all_data.extend(page_data)

            logging.info(f"✅ Finished {os.path.basename(pdf_path)}, found {len(all_data)} records.")
            return all_data

        except ImportError:
            logging.error("❌ Missing dependencies. Install: pip install pdf2image google-cloud-vision pandas")
            return []
        except Exception as e:
            logging.error(f"❌ An unexpected error occurred while processing {pdf_path}: {e}", exc_info=True)
            return []


# -----------------------------
# Dask helpers
# -----------------------------

def _process_pdf_worker(args) -> List[Dict]:
    pdf_path, credentials_path, template = args
    extractor = PDFTableExtractor(credentials_path)
    if not extractor.vision_client:
        return []
    extractor.table_template = template
    return extractor.extract_data_from_pdf(pdf_path, create_template=False)


def _save_to_csv(data: List[Dict], output_path: str) -> None:
    if not data:
        logging.warning("❌ No data extracted.")
        return
    try:
        df = pd.DataFrame(data)[['filename', 'epic_no', 'age', 'gender', 'reason']]
        df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"🎉 SUCCESS! Extracted {len(df)} rows → {output_path}")
        print("\n" + "="*20 + " SAMPLE DATA " + "="*20)
        print(df.head(15))
        print("="*53)
    except Exception as e:
        logging.error(f"❌ Failed to save data to CSV: {e}")


def main(ac_name):
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "./cad-de-itp-e06303b7a0f2.json")
    output_path = "./output/"+ac_name+"_combined_extraction_output.csv"

    pdf_files = sorted(glob.glob("./inputs/"+ac_name+'/*.pdf'))
    if not pdf_files:
        logging.warning("No PDF files found in the current directory.")
        return

    logging.info(f"Found {len(pdf_files)} PDFs. Building template from: {pdf_files[0]}")
    bootstrap_extractor = PDFTableExtractor(credentials_path)
    if not bootstrap_extractor.vision_client:
        return

    data_first = bootstrap_extractor.extract_data_from_pdf(
        pdf_files[0], create_template=True, create_debug_file=True
    )

    template = bootstrap_extractor.table_template
    if not template:
        logging.error("No template could be built from the first PDF. Exiting.")
        return

    all_data: List[Dict] = []
    if data_first:
        all_data.extend(data_first)

    remaining = pdf_files[1:]
    if not remaining:
        _save_to_csv(all_data, output_path)
        return

    n_workers = int(os.environ.get("DASK_WORKERS", str(max(1, min(os.cpu_count() or 8, 12)))))
    n_threads = int(os.environ.get("DASK_THREADS", "4"))
    logging.info(f"Spinning up LocalCluster with {n_workers} workers × {n_threads} threads each.")

    with LocalCluster(n_workers=n_workers, threads_per_worker=n_threads, processes=True) as cluster, Client(cluster) as client:
        bag = db.from_sequence([(p, credentials_path, template) for p in remaining], npartitions=n_workers)
        results = bag.map(_process_pdf_worker).compute()

    for chunk in results:
        if chunk:
            all_data.extend(chunk)

    _save_to_csv(all_data, output_path)

###############################
# 1. Change n_workers and n_threads based on your computer
# 2. Install UV `pip install uv`  more at https://docs.astral.sh/uv/getting-started/installation/
# 3. Then run `uv run gender_extract_ge_all.py`
# 4. This uses on big computer but you can also distribute across the LAN esily
###############################
if __name__ == "__main__":
    loop_start = timer()

    # complete
    ac_name_AC1  =  ["AC1","AC2","AC3","AC4","AC5","AC6","AC7","AC8","AC9","AC10"]
    ac_name_AC11 = ["AC11","AC12","AC13","AC14","AC15","AC16","AC17","AC18","AC19","AC20"]
    ac_name_AC21 = ["AC21","AC22","AC23","AC24","AC25","AC26","AC27","AC28","AC29", "AC30"]    
    ac_name_AC31 = ["AC31","AC32","AC33","AC34","AC35","AC36","AC37","AC38","AC39", "AC40"]
    ac_name_AC41 = ["AC41","AC42","AC43","AC44","AC45","AC46","AC47","AC48","AC49","AC50"]
    ac_name_AC51 = ["AC51","AC52","AC53","AC54","AC55","AC56","AC57","AC58","AC59","AC60"]
    ac_name_AC61 = ["AC61","AC62","AC63","AC64","AC65","AC66","AC67","AC68","AC69", "AC70"]
    ac_name_AC71 = ["AC71","AC72","AC73","AC74","AC75","AC76","AC77","AC78","AC79", "AC80"]
    ac_name_AC81 = ["AC81","AC82","AC83","AC84","AC85","AC86","AC87","AC88","AC89", "AC90"]
    ac_name_AC91 = ["AC91","AC92","AC93","AC94","AC95","AC96","AC97","AC98","AC99","AC100"]
    ac_name_AC101 = ["AC101","AC102","AC103","AC107","AC104","AC105","AC106","AC108","AC109","AC110"]
    ac_name_AC111 = ["AC111","AC112","AC113","AC114","AC115","AC116","AC117","AC118","AC119","AC120"]
    ac_name_AC121 = ["AC121","AC122","AC123","AC124","AC125","AC126","AC127","AC128","AC129","AC130"]
    ac_name_AC131 = ["AC131","AC132","AC133","AC134","AC135","AC136","AC137","AC138","AC139","AC140"]
    ac_name_AC141 = ["AC141","AC142","AC143","AC144","AC145","AC146","AC147","AC148","AC149","AC150"]    
    ac_name_AC151 = ["AC151","AC152","AC153","AC154","AC155","AC156","AC157","AC158","AC159","AC160"]
    ac_name_AC161 = ["AC161","AC162","AC163","AC164","AC165","AC166","AC167","AC168","AC169","AC170"]
    ac_name_AC171 = ["AC171","AC172","AC173","AC174","AC175","AC176","AC177","AC178","AC179","AC180"]
    ac_name_AC181 = ["AC181","AC182","AC183","AC184","AC185","AC186","AC187","AC189","AC188","AC190"] 
    ac_name_AC191 = ["AC191","AC192","AC193","AC194","AC195","AC196","AC197","AC198","AC199","AC200"]
    ac_name_AC201 = ["AC201","AC202","AC203","AC204","AC205","AC206","AC207","AC208","AC209","AC210"]
    ac_name_AC211 = ["AC211","AC212","AC213","AC214","AC215","AC216","AC217","AC218","AC219","AC220"]
    ac_name_AC221 = ["AC221","AC222","AC223","AC224","AC225","AC226","AC227","AC228","AC229","AC230"]
    ac_name_AC231 = ["AC231","AC232","AC233","AC234","AC235","AC236","AC237","AC238","AC239","AC240"]
    ac_name_AC241 = ["AC241","AC242","AC243","AC244"]


    # to run, change the batch
    ac_names = ac_name_AC101
    print(ac_names)
    for ac_name in ac_names:
        print("Running for ac_name = " + ac_name)
        start = timer()
        main(ac_name)
        end = timer()
        print("================== time taken in minutes =========================")
        print(start)
        print(end)
        print(timedelta(seconds=end-start))
        print("Elapsed minutes:", (end - start) / 60)
    loop_end = timer()
    print("Loop Elapsed minutes:", (loop_end - loop_start) / 60)
