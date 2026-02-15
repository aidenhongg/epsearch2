from dotenv import load_dotenv

load_dotenv()

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, LogitsProcessorList, LogitsProcessor, BitsAndBytesConfig
from pathlib import Path
from PIL import Image
import csv
import io
import torch
import fitz

MODEL_LIB = r"D:\AI_MODELS\QWEN"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
TARGETS = {"text": 0, "blank": 1, "other": 2}
PROMPT = """
Classify the provided image into exactly one of these categories:

- text: Scans of text documents.

- blank: Empty pages, or scans that are entirely black or white.

- other: Photographs, drawings, or any document not fitting the above.

Output Requirement: Return only the category name.
""".strip()

CORPUS_PATH = Path("./corpus")
BAD_LOG = Path("./bad_pdfs.txt")

BATCH_SIZE = 5

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto",
    cache_dir=MODEL_LIB,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_LIB
)

ids_to_bias = {processor.tokenizer.eos_token_id} # Always allow the model to stop
for word in TARGETS.keys():
    # Capture: "word", "Word", " word", " Word"
    for variant in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
        tokens = processor.tokenizer.encode(variant, add_special_tokens=False)
        ids_to_bias.update(tokens)

BIAS_IDS = torch.tensor(list(ids_to_bias))

# Precompute the chat template text once (it's identical for every image)
_TEMPLATE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "placeholder"},
            {"type": "text", "text": PROMPT},
        ],
    }
]
TEXT_PROMPT = processor.apply_chat_template(_TEMPLATE_MESSAGES, tokenize=False, add_generation_prompt=True)

class TargetTokenBiasProcessor(LogitsProcessor):
    def __init__(self, allowed_ids_tensor, device):
        self.allowed_ids = allowed_ids_tensor.to(device)
    
    def __call__(self, _, scores):
        # scores shape is [batch_size, vocab_size]
        mask = torch.full_like(scores, float("-inf"))
        
        # Vectorized operation on the GPU
        mask[:, self.allowed_ids] = 0
        
        return scores + mask

# Precompute the logits processor list (same for every call)
LOGITS_PROCESSORS = LogitsProcessorList([TargetTokenBiasProcessor(BIAS_IDS, model.device)])

def classify_image(image) -> str:
    # Pass the PIL image directly — no need for process_vision_info
    inputs = processor(
            text=[TEXT_PROMPT],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            logits_processor=LOGITS_PROCESSORS,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return output_text[0].strip().lower()


def classify_images_batch(images: list) -> list[str]:
    """Classify a batch of PIL images. Returns a list of classification strings."""
    inputs = processor(
        text=[TEXT_PROMPT] * len(images),
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            logits_processor=LOGITS_PROCESSORS,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return [t.strip().lower() for t in output_texts]


def yield_images():
    for i in range(7, 8):
        pdf_path = CORPUS_PATH / str(i) / "pdfs"
        scans_path = CORPUS_PATH / str(i) / "scans"
        classifications_path = CORPUS_PATH / str(i) / "classifications.csv"
        if not scans_path.exists():
            scans_path.mkdir(parents=True)

        if not classifications_path.exists():
            with open(classifications_path, "w", newline="") as f:
                csv.writer(f).writerow(["scan_path", "filename", "page", "image", "class"])

        with open(classifications_path, "a", newline="") as clf_file:
            clf_writer = csv.writer(clf_file)
            for pdf in pdf_path.glob('*.pdf'): # loop through pdfs
                with fitz.open(pdf) as doc:
                
                    for page_index in range(len(doc)): # loop through pages in each pdf
                        image_list = doc.get_page_images(page_index)
                        for img_index, img_info in enumerate(image_list): # loop through images in page
                            xref = img_info[0]
                            image_data = doc.extract_image(xref)["image"]
                            yield (scans_path, pdf, clf_file, clf_writer, page_index, img_index, image_data)

def _flush_batch(batch_images, batch_meta, clf_writer, clf_file, bad_writer, bad_file):
    """Classify a collected batch and write results."""
    try:
        results = classify_images_batch(batch_images)
    except Exception as e:
        # If the whole batch fails, log every item
        for scans_path, pdf_file, page_index, img_index, _ in batch_meta:
            print(e)
            bad_writer.writerow([str(pdf_file), page_index, img_index, str(e)])
        bad_file.flush()
        return

    for (scans_path, pdf_file, page_index, img_index, pil_img), result in zip(batch_meta, results):
        try:
            if result not in TARGETS:
                # Retry individually up to 3 more times
                for _ in range(3):
                    result = classify_image(pil_img)
                    if result in TARGETS:
                        break
                else:
                    raise Exception("Failed to classify after batch + 3 retries.")

            out_path = scans_path / f"{pdf_file.stem}_p{page_index}_i{img_index}.png"
            pil_img.save(out_path, format="PNG")
            clf_writer.writerow([str(out_path), f"{pdf_file.stem}", page_index, img_index, TARGETS[result]])
            clf_file.flush()
        except Exception as e:
            print(e)
            bad_writer.writerow([str(pdf_file), page_index, img_index, str(e)])
            bad_file.flush()


def main():
    if not BAD_LOG.exists():
        with open(BAD_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["pdf", "page", "image", "error"])
    
    with open(BAD_LOG, "a", newline="") as bad_file:
        bad_writer = csv.writer(bad_file)

        batch_meta = []   # (scans_path, pdf_file, page_index, img_index)
        batch_images = []  # PIL images
        

        for scans_path, pdf_file, clf_file, clf_writer, page_index, img_index, image_data in yield_images():
            try:
                pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(e)
                bad_writer.writerow([str(pdf_file), page_index, img_index, str(e)])
                bad_file.flush()
                continue

            batch_meta.append((scans_path, pdf_file, page_index, img_index, pil_img))
            batch_images.append(pil_img)

            if len(batch_images) < BATCH_SIZE:
                continue

            _flush_batch(batch_images, batch_meta, clf_writer, clf_file, bad_writer, bad_file)
            batch_meta.clear()
            batch_images.clear()

        # Process any remaining images
        if batch_images:
            _flush_batch(batch_images, batch_meta, clf_writer, clf_file, bad_writer, bad_file)


if __name__ == "__main__":
    main()
