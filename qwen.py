import os
import time
from PIL import Image

from transformers import AutoTokenizer  # noqa: F401 (tokenizer kept for future use)
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

# ---------------------------------------------------------------------------
# Default hyper‑parameters that can be overridden via CLI flags
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"  # single‑GPU, 7‑B variant
DEFAULT_TP    = 1                               # tensor‑parallel size

# Prevent PIL from crashing on very large images
Image.MAX_IMAGE_PIXELS = 933_120_000

# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

def run_qwen2_5_vl(question: str,
                   modality: str,
                   model_name: str = DEFAULT_MODEL,
                   tp_size: int = DEFAULT_TP,
                   disable_cache: bool = False):
    """Create an LLM instance, build the system+user prompt, return all parts."""

    llm = LLM(
        model=model_name,
        max_model_len=12_000,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1_280 * 28 * 28,
            "fps": 1,
        },
        tensor_parallel_size=tp_size,
        disable_mm_preprocessor_cache=disable_cache,
    )

    placeholder = "<|image_pad|>" if modality == "image" else "<|video_pad|>"

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Qwen 2.5 models do not require a special stop token; pass None
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def get_multi_modal_input(args):
    """Prepare image or video asset + task‑specific question."""

    if args.modality == "image":
        image = Image.open(
            "./qwen_testing/22-04-2016-p-/22-04-2016-p-_page_1.png"
        ).convert("RGB")

        if args.output_type == "html":
            img_question = (
                """
                You are tasked with converting a scanned document image into a fully‑renderable HTML document. The image may contain complicated tables, handwritten elements, stamps, logos, and other details. Your goal is to create an accurate HTML representation of the document using only the information visible in the image.

                Follow these steps to create the HTML document:
                1. **Analyze** the image content: identify sections, tables, headings, etc.
                2. **Create** the basic HTML skeleton with <!DOCTYPE html>.
                3. **Represent** the content using proper tags; use <span class="handwritten"> for handwriting, <div class="stamp"> for stamps, etc.
                4. **Style** via an inline <style> block so the rendered page mimics the original.
                5. **Review** that every visible element is captured.
                6. Output the full HTML starting with <!DOCTYPE html> and ending with </html>.
                """
            )
        else:  # doc_type
            img_question = (
                """
                Examine the document image and state **only** its type, choosing **exactly one** from:
                Circulars, Notifications, Orders, Memorandums, Gazettes, Acts/Rules, Policies, Resolutions, Guidelines, Licenses, Minutes of Meetings, Forms, Receipts, Bonds, Tenders, Resumes, Bio‑data.
                Reply with the chosen type *plain text* – no explanation.
                """
            )

        return {"data": image, "question": img_question}

    if args.modality == "video":
        video = VideoAsset(name="sample_demo_1.mp4", num_frames=args.num_frames).np_ndarrays
        return {"data": video, "question": "Why is this video funny?"}

    raise ValueError(f"Unsupported modality: {args.modality}")


def apply_image_repeat(folder_path: str, prompt: str, modality: str):
    """Load every image in *folder_path* once."""
    inputs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(os.path.join(folder_path, file)).convert("RGB")
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {modality: image},
                    "image_name": file,
                })
            except Exception as exc:  # keep going on bad files
                print(f"[WARN] Could not load {file}: {exc}")
    return inputs


def folders_pdf(folder_path: str, prompt: str, modality: str):
    """Recurse one level down (folder/IMG) and collect images."""
    inputs = []
    for img_folder in os.listdir(folder_path):
        sub_dir = os.path.join(folder_path, img_folder)
        if not os.path.isdir(sub_dir):
            continue
        for img_file in os.listdir(sub_dir):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    image = Image.open(os.path.join(sub_dir, img_file)).convert("RGB")
                    inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {modality: image},
                        "image_name": img_file,
                    })
                except Exception as exc:
                    print(f"[WARN] Could not load {img_file}: {exc}")
    return inputs


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

model_example_map = {
    "qwen2_5_vl": run_qwen2_5_vl,
}


def main(args):
    if args.model_type not in model_example_map:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # 1) Build the single prompt
    mm_input = get_multi_modal_input(args)
    question = mm_input["question"]
    modality = args.modality

    llm, prompt, stop_ids = model_example_map[args.model_type](
        question,
        modality,
        model_name=args.hf_model_name,
        tp_size=args.tp_size,
        disable_cache=args.disable_mm_preprocessor_cache,
    )

    # 2) Prepare data list
    if args.type_of_input == "images":
        inputs = apply_image_repeat(args.input_folder_path, prompt, modality)
    else:  # subfolder_images
        inputs = folders_pdf(args.input_folder_path, prompt, modality)

    if not inputs:
        raise RuntimeError("No images found for inference — check your paths.")

    # 3) Sampling parameters and generate loop
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        max_tokens=20,
        stop_token_ids=stop_ids,
    )

    start = time.time()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    if args.time_generate:
        print(f"generate() took {time.time() - start:.2f}s for {len(inputs)} prompts")

    # For now just pretty‑print the first output
    print("====== SAMPLE OUTPUT ======")
    print(outputs[0].outputs[0].text if outputs else "<no output>")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="vLLM Qwen‑VL offline demo")

    parser.add_argument("--model-type", "-m", choices=model_example_map.keys(), default="qwen2_5_vl")
    parser.add_argument("--hf-model-name", default=DEFAULT_MODEL, help="HF repo id to load (default 7‑B)")
    parser.add_argument("--tp-size", type=int, default=DEFAULT_TP, help="Tensor‑parallel size; 1 = single GPU")

    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--modality", choices=["image", "video"], default="image")
    parser.add_argument("--num-frames", type=int, default=16)

    parser.add_argument("--disable-mm-preprocessor-cache", action="store_true")
    parser.add_argument("--time-generate", action="store_true")

    parser.add_argument("--type-of-input", choices=["images", "subfolder_images"], default="subfolder_images")
    parser.add_argument("--input-folder-path", default="./qwen_testing/")
    parser.add_argument("--output-html-dir", default="./outputs")
    parser.add_argument("--output-type", choices=["html", "doc_type"], default="html")

    args = parser.parse_args()
    main(args)
