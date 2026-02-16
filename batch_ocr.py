from dotenv import load_dotenv

load_dotenv()

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from PIL import Image
import csv
import json
import torch
torch.cuda.empty_cache()
from datetime import datetime


MODEL_LIB = r"./qwen"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

PROMPT = """
Transcribe the provided image into GitHub Flavored Markdown. 

### STRICT TRANSCRIPTION RULES:
1. **Verbatim Accuracy:** Transcribe all text EXACTLY as it appears. Do NOT correct spelling, punctuation, or grammar. 
2. **Spacing & Typos:** If a word has improper spaces (e.g., "l e g a l") or typos, preserve them exactly.
3. **Redactions:** Represent redacted text or black bars using [REDACTED] or "█".
4. **Structure:** - Convert visual tables into Markdown table syntax. 
   - Maintain headers (#, ##), bullet points, and numbered lists.
5. **Formatting:** Use bolding or italics only where it exists in the original text.

### OUTPUT INSTRUCTIONS:
- Provide ONLY the Markdown code. 
- No preamble, no postscript, and no conversational filler.
""".strip()

CORPUS_PATH = Path("./corpus")
BAD_LOG = Path("./bad_OCR.txt")
BATCH_SIZE = 16
NUM_LOAD_WORKERS = 4

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    cache_dir=MODEL_LIB,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_LIB
)

_TEMPLATE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "placeholder"},
            {"type": "text", "text": PROMPT},
        ],
    }
]
TEXT_PROMPT = processor.apply_chat_template(
    _TEMPLATE_MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=False
)


def _load_and_resize(scan_path: Path) -> Image.Image:
    img = Image.open(scan_path).convert("RGB")
    return img


def delete_md(output_text: str) -> str:
    if output_text.startswith("```"):
        output_text = output_text[3:].strip()
    if output_text.endswith("```"):
        output_text = output_text[:-3].strip()
    if output_text.lower().startswith("markdown"):
        output_text = output_text[8:].lstrip()
    elif output_text.lower().startswith("md"):
        output_text = output_text[2:].lstrip()
    
    return output_text.strip()


def _preprocess_batch(images: list[Image.Image]):
    """CPU-bound: tokenize images and prepare model-ready tensors."""
    inputs = processor(
        text=[TEXT_PROMPT] * len(images),
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    return inputs


def _generate_batch(inputs) -> list[str]:
    """GPU-bound: run inference and decode output tokens."""
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    del generated_ids
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    del generated_ids_trimmed

    return [delete_md(t) for t in output_texts]


def _load_images_parallel(batch_meta, bad_writer, bad_file):
    """Load and resize images in parallel threads. Returns (images, valid_meta) excluding failures."""
    loaded_images = []
    valid_meta = []

    with ThreadPoolExecutor(max_workers=NUM_LOAD_WORKERS) as pool:
        futures = [
            (pool.submit(_load_and_resize, scan_path), (scan_path, filename, page, img_index))
            for scan_path, filename, page, img_index in batch_meta
        ]
        for future, meta in futures:
            try:
                img = future.result()
                loaded_images.append(img)
                valid_meta.append(meta)
            except Exception as e:
                scan_path, filename, page, img_index = meta
                print(e)
                bad_writer.writerow([str(scan_path), filename, page, img_index, str(e)])
                bad_file.flush()

    return loaded_images, valid_meta


class OCRPipeline:
    """Runs OCR (preprocess + GPU inference) in a background thread so the main
    thread can load the next batch of images concurrently."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: Future | None = None

    def submit(self, images, meta, record, bad_writer, bad_file):
        """Submit a batch for OCR. Blocks until the previous batch completes."""
        self.flush()
        self._pending = self._executor.submit(
            self._process_batch, images, meta, record, bad_writer, bad_file
        )

    @staticmethod
    def _process_batch(images, meta, record, bad_writer, bad_file):
        try:
            inputs = _preprocess_batch(images)
            del images
            results = _generate_batch(inputs)
            del inputs
            torch.cuda.empty_cache()
            for (scan_path, filename, page, img_index), text in zip(meta, results):
                key = f"p{page}i{img_index}"
                record[key] = {"text": text}
        except Exception as e:
            for scan_path, filename, page, img_index in meta:
                print(e)
                bad_writer.writerow([str(scan_path), filename, page, img_index, str(e)])
            bad_file.flush()

    def flush(self):
        """Block until any pending OCR work completes."""
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def shutdown(self):
        self.flush()
        self._executor.shutdown(wait=False)


def yield_images():
    """Stream classifications.csv rows where class==0, yielding (corpus_index, scan_path, filename, page, image)."""
    for i in range(7, 8):
        classifications_path = CORPUS_PATH / str(i) / "classifications.csv"
        if not classifications_path.exists():
            continue
        with open(classifications_path, "r", newline="") as clf_file:
            reader = csv.DictReader(clf_file)
            for row in reader:
                if row["class"] == "0":
                    yield (i, Path(row["scan_path"].replace("\\", "/")), row["filename"], int(row["page"]), int(row["image"]))

def _write_record(ocr_file, record):
    """Write a completed filename record to JSONL if it has any OCR entries."""
    if len(record) > 1:  # more than just the "filename" key
        ocr_file.write(json.dumps(record) + "\n")
        ocr_file.flush()


def main():
    if not BAD_LOG.exists():
        with open(BAD_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["scan_path", "filename", "page", "image", "error"])

    pipeline = OCRPipeline()

    with open(BAD_LOG, "a", newline="") as bad_file:
        bad_writer = csv.writer(bad_file)

        current_corpus = None
        current_filename = None
        current_record = {}
        ocr_file = None
        batch_meta = []  # (scan_path, filename, page, img_index) — images loaded later in parallel

        bad_writer.writerow([str(datetime.now()),'','','',''])
        bad_file.flush()
        try:
            for corpus_idx, scan_path, filename, page, img_index in yield_images():
                # Open new JSONL file when corpus dir changes
                if corpus_idx != current_corpus:
                    if batch_meta:
                        images, valid = _load_images_parallel(batch_meta, bad_writer, bad_file)
                        if images:
                            pipeline.submit(images, valid, current_record, bad_writer, bad_file)
                        batch_meta.clear()
                    pipeline.flush()
                    if current_record:
                        _write_record(ocr_file, current_record)
                        current_record = {}
                        current_filename = None
                    if ocr_file:
                        ocr_file.close()
                    ocr_path = CORPUS_PATH / str(corpus_idx) / "OCR.jsonl"
                    ocr_file = open(ocr_path, "a", encoding="utf-8")
                    current_corpus = corpus_idx

                # When filename changes, flush everything then write completed record
                if filename != current_filename:
                    if batch_meta:
                        images, valid = _load_images_parallel(batch_meta, bad_writer, bad_file)
                        if images:
                            pipeline.submit(images, valid, current_record, bad_writer, bad_file)
                        batch_meta.clear()
                    pipeline.flush()
                    if current_record:
                        _write_record(ocr_file, current_record)
                    current_record = {"filename": filename}
                    current_filename = filename

                batch_meta.append((scan_path, filename, page, img_index))

                if len(batch_meta) < BATCH_SIZE:
                    continue

                # Batch is full — load images in parallel, then submit to pipeline
                images, valid = _load_images_parallel(batch_meta, bad_writer, bad_file)
                batch_meta.clear()
                if images:
                    pipeline.submit(images, valid, current_record, bad_writer, bad_file)

            # Process remaining images and write final record
            if batch_meta:
                images, valid = _load_images_parallel(batch_meta, bad_writer, bad_file)
                if images:
                    pipeline.submit(images, valid, current_record, bad_writer, bad_file)
            pipeline.flush()
            if current_record:
                _write_record(ocr_file, current_record)
        finally:
            bad_writer.writerow([str(datetime.now()),'','','',''])
            pipeline.shutdown()
            if ocr_file:
                ocr_file.close()


if __name__ == "__main__":
    main()