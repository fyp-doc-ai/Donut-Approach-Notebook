Sinhala Document Understanding: End-to-End Extraction using Donut
📌 Project Objective
The primary objective of this project is to accurately identify and extract structured information (such as questions, answers, and headers) from Sinhala administrative forms containing both handwritten and printed text. This project transitions from a legacy, bounding-box-dependent OCR pipeline to a modern, OCR-free Vision-Language architecture to improve accuracy and robustness on complex, real-world document layouts.

🚀 Research Contribution
Better Sinhala document extraction (End-to-end Donut)
This research replaces the current multi-stage pipeline (TrOCR + LiLT + XLM-R) with an OCR-free, end-to-end Vision-Language Model (Donut) adapted for Sinhala. By doing text recognition and layout understanding simultaneously in one model, it avoids cascading OCR errors. The study will measure the accuracy gain and structural extraction improvements on the SinFUND dataset.

📂 Data Formation Approach
Unlike traditional pipeline models (like LiLT), the Donut architecture does not require or consume bounding box coordinates (X, Y pixels).

The dataset generation script (generate_donut_data.py) converts the existing FUNSD-style JSON annotations into a format compatible with Donut:

Removes Spatial Data: Drops all bounding box coordinates and ID tags.

Structures the Ground Truth: Extracts only the semantic labels (e.g., question, answer) and the raw text.

Creates metadata.jsonl: Pairs the raw document image with a stringified JSON sequence of the extracted entities, placing them in a unified Hugging Face imagefolder format.

💻 Current Setup (donut-doc-understanding.ipynb)
The primary training notebook facilitates two architectural options:

Option A (Default): Utilizing the base Donut Swin-Transformer and its default BART decoder.

Option B (Hybrid/Lego Approach): Merging the full-page layout understanding of Donut's Swin Vision Encoder with the Sinhala linguistic mastery of a pre-trained TrOCR SinBERT text decoder.

⚠️ Known Issue: The "0% Validation Accuracy" Problem
Currently, during the validation and evaluation phases, the model outputs exactly 0 for all metrics (Precision, Recall, F1-score).

Root Cause Analysis:
This is not a failure of the model's vision capabilities, but rather a "Domain Shift" tokenization failure.

Donut's architecture relies on generating XML-like structural tags (e.g., <s_label>, <s_text>) to separate entities, not raw JSON brackets like { or }.

Because these specific structural tokens were not explicitly added to the tokenizer's vocabulary, the model views the JSON structure as an unknown language. It attempts to split the structure into random sub-word tokens, resulting in malformed, hallucinated syntax during text generation.

During the evaluation loop, the script uses a function (processor.token2json) to convert the generated text back into a Python dictionary to compare against the ground truth. Because the generated syntax is broken and missing proper XML tags, the parser fails entirely, defaulting to an empty extraction and triggering a 0% score across all metrics.

## 🔧 Status Update (2026-08-30 — Session 1)

A deeper investigation (see `Guide/learnings.md`) found that the tokenization issue above is real but is only part of the story. The bigger problem: the dataset's ground truth uses the literal, highly variable Sinhala form-field label text as the JSON *key* itself (e.g. `{"3 දිස්ත්‍රික්කය": "කලුතර", ...}`). Donut's `json2token` turns every JSON key into its own XML tag, so this would require the model to generate a near-unique special token per document — something ~100 examples spread across many form layouts cannot teach it. This is a data-schema problem, not just a missing-special-tokens problem.

`donut-doc-understanding.ipynb` has been rewritten around a fix: the ground truth is reshaped in-memory into Donut's own schema convention — a small, fixed set of tags (`form` / `question` / `answer`) with the variable Sinhala text as tag *content* rather than tag *names*, mirroring how Donut's official CORD/FUNSD fine-tuning tasks are built. Several other concrete bugs were fixed alongside it (wrong `decoder_start_token_id`, missing `generation_max_length` causing truncated evaluation generations, inconsistent/unvalidated `MAX_LENGTH`, too few optimizer steps for the dataset size, a likely-incorrect hardcoded Kaggle dataset path, and a broken regex-based evaluation parser). Full details and rationale are in `Guide/learnings.md`; the Kaggle-compatibility rules this notebook must keep following are in `Guide/rules.md`.

**Model variant options** (`MODEL_VARIANT` in Cell 1 of the notebook) now cover three configurations instead of two:

- `donut_native` — Donut Swin encoder + Donut's own decoder + Donut's original tokenizer. Fast sanity check only (Sinhala text hits inefficient byte-level tokenization).
- `donut_sinhala_vocab` (**new recommended default**) — Donut Swin encoder + Donut's own decoder (keeping its pretrained cross-attention) + SinBERT tokenizer for Sinhala-efficient vocabulary. This refines "Option A" below.
- `hybrid` — Donut Swin encoder + custom Sinhala TrOCR decoder + SinBERT tokenizer. This is "Option B" below — the original "Lego" research contribution. Kept for the planned ablation, but now understood to carry real architectural risk (its cross-attention was never jointly pretrained with Donut's encoder), so it should only be evaluated after `donut_sinhala_vocab` produces sane, non-zero validation metrics.

None of these fixes have been confirmed by an actual Kaggle run yet as of this update — this environment has no local GPU or ML libraries, so changes are reasoned from static analysis of the code and dataset, not execution. See `Guide/whatIsNext.md` for the current handoff status.