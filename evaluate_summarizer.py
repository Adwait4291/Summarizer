import os
import torch
import nltk
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments, # Still useful for setting generation params
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import PeftModel, PeftConfig
import evaluate
import logging
from tqdm import tqdm
from functools import partial

# --- Configuration ---
# *** Ensure this matches the OUTPUT_DIR from the training script ***
MODEL_PATH = "../saved_model/fine_tuned_summarizer_bart_custom"
# Base model needed if loading LoRA adapter - MUST match training
BASE_MODEL_NAME = "facebook/bart-large-cnn"
DATA_DIR = "../data"
TEST_FILE = os.path.join(DATA_DIR, 'test.csv') # Assumes test.csv exists in the extracted data

# *** IMPORTANT: Update these based on the actual column names in your CSV files ***
# *** Must match the columns used during training ***
DOCUMENT_COLUMN = "dialogue" # Or 'text', 'article', etc. - Check your test.csv
SUMMARY_COLUMN = "summary"   # Or 'headlines', etc. - Check your test.csv
# PREFIX = "summarize: " # Removed: BART doesn't need it

# Tokenization & Generation Parameters - MUST match or be compatible with training
MAX_INPUT_LENGTH = 1024 # Should match training tokenizer settings
MAX_TARGET_LENGTH = 128 # Max length for generated summaries during evaluation
BATCH_SIZE = 8 # Adjust based on GPU memory for generation
GENERATION_NUM_BEAMS = 4

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load ROUGE Metric ---
# (Using the same robust loading logic as in the training script)
try:
    rouge_metric = evaluate.load("rouge")
    nltk.download('punkt', quiet=True)
    logger.info("ROUGE metric loaded successfully.")
except Exception as e:
    try:
        logger.warning(f"Failed loading ROUGE from hub: {e}. Trying local path.")
        import site
        import sys
        script_path = None
        for path in site.getsitepackages():
             potential_path = os.path.join(path, 'evaluate', 'metrics', 'rouge', 'rouge.py')
             if os.path.exists(potential_path):
                 script_path = potential_path
                 break
        if not script_path:
             potential_path = os.path.join(site.getusersitepackages(), 'evaluate', 'metrics', 'rouge', 'rouge.py')
             if os.path.exists(potential_path):
                 script_path = potential_path
        if script_path:
             logger.info(f"Found rouge script at: {script_path}")
             rouge_metric = evaluate.load(script_path)
             logger.info("ROUGE metric loaded successfully from local path.")
        else:
             raise FileNotFoundError("Could not find rouge.py script locally.")
    except Exception as local_e:
        logger.error(f"Error loading ROUGE metric locally: {local_e}")
        logger.error("Please ensure 'evaluate', 'rouge_score', and 'nltk' are installed. Check for conflicting local 'rouge' folders.")
        rouge_metric = None


# --- Main Evaluation Function ---
def evaluate_summarizer():
    logger.info(f"Starting evaluation process for BART summarizer model at: {MODEL_PATH}")

    # --- Check if test data exists ---
    if not os.path.exists(TEST_FILE):
        logger.error(f"Test file not found at {TEST_FILE}")
        logger.error(f"Please ensure you have downloaded, extracted, and placed test.csv from the dataset into the '{DATA_DIR}' directory.")
        # Create dummy file for basic script execution, but evaluation will be meaningless
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.warning("Creating dummy test.csv for script execution. Evaluation results will not be meaningful.")
        with open(TEST_FILE, 'w') as f:
            # Use the configured column names for the dummy file
            f.write(f"{DOCUMENT_COLUMN},{SUMMARY_COLUMN}\n")
            f.write(f"\"This is dummy document content.\",\"This is a dummy summary.\"\n")
        # return # Optionally exit if real test data is mandatory

    # --- 1. Load Tokenizer ---
    logger.info(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        # Check if MODEL_PATH exists before loading
        if not os.path.isdir(MODEL_PATH):
             logger.error(f"Model directory not found: {MODEL_PATH}")
             logger.error("Please ensure the training script ran successfully and saved the model to the correct location.")
             return
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        logger.error(f"Error loading tokenizer from {MODEL_PATH}: {e}", exc_info=True)
        return

    # --- 2. Load Model ---
    logger.info("Loading fine-tuned model...")
    try:
        is_peft_model = os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if is_peft_model:
            logger.info("Detected PEFT/LoRA model. Loading base model and attaching adapter...")
            config = PeftConfig.from_pretrained(MODEL_PATH)
            base_model_name_or_path = config.base_model_name_or_path
            if not base_model_name_or_path:
                 logger.warning(f"PEFT config does not specify base model. Using default: {BASE_MODEL_NAME}")
                 base_model_name_or_path = BASE_MODEL_NAME
            elif base_model_name_or_path != BASE_MODEL_NAME:
                 logger.warning(f"PEFT config base model '{base_model_name_or_path}' differs from specified '{BASE_MODEL_NAME}'. Using config's value.")

            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name_or_path)
            model = PeftModel.from_pretrained(base_model, MODEL_PATH)
            # model = model.merge_and_unload() # Optional: merge for faster inference
            logger.info("PEFT model loaded.")
        else:
            logger.info("Detected full fine-tuned model. Loading directly...")
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
            logger.info("Full model loaded.")

        model.eval()
        model.to(device)
        logger.info(f"Model moved to device: {device}")
        logger.info(f"Model's configured max positional embeddings: {model.config.max_position_embeddings}")

    except Exception as e:
        logger.error(f"Error loading model from {MODEL_PATH}: {e}", exc_info=True)
        return

    # --- 3. Load Test Dataset ---
    logger.info(f"Loading test dataset: {TEST_FILE}")
    try:
        test_dataset = load_dataset('csv', data_files={'test': TEST_FILE})['test']
        # Check if specified columns exist
        if DOCUMENT_COLUMN not in test_dataset.column_names:
             raise ValueError(f"Document column '{DOCUMENT_COLUMN}' not found in {TEST_FILE}. Actual columns: {test_dataset.column_names}. Please update DOCUMENT_COLUMN variable.")
        if SUMMARY_COLUMN not in test_dataset.column_names:
             raise ValueError(f"Summary column '{SUMMARY_COLUMN}' not found in {TEST_FILE}. Actual columns: {test_dataset.column_names}. Please update SUMMARY_COLUMN variable.")

        reference_summaries = test_dataset[SUMMARY_COLUMN]
        # Handle potential None values in reference summaries
        reference_summaries = ["" if text is None else str(text) for text in reference_summaries]

        logger.info(f"Test dataset loaded. Size: {len(test_dataset)}")
    except Exception as e:
        logger.error(f"Error loading or processing test dataset: {e}", exc_info=True)
        return

    # --- 4. Generate Summaries ---
    logger.info(f"Generating summaries (max input: {MAX_INPUT_LENGTH}, max output: {MAX_TARGET_LENGTH})...")
    generated_summaries = []
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    for batch in tqdm(dataloader, desc="Generating Summaries"):
        # Prepare inputs for the model
        # Handle potential None values in input documents
        inputs = ["" if text is None else str(text) for text in batch[DOCUMENT_COLUMN]]

        inputs_tokenized = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="longest", # Pad batch to longest sequence
            return_tensors="pt"
        ).to(device)

        # Generate
        try:
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs_tokenized["input_ids"],
                    attention_mask=inputs_tokenized["attention_mask"],
                    max_length=MAX_TARGET_LENGTH + 2, # Add buffer
                    num_beams=GENERATION_NUM_BEAMS,
                    early_stopping=True
                )
        except Exception as e:
            logger.error(f"Error during model.generate: {e}", exc_info=True)
            # Add placeholder summaries to avoid crashing ROUGE calculation if possible
            batch_summaries = ["[GENERATION ERROR]" for _ in range(len(inputs))]
            generated_summaries.extend(batch_summaries)
            continue # Skip decoding for this batch

        # Decode generated IDs
        batch_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_summaries.extend(batch_summaries)

    logger.info(f"Generated {len(generated_summaries)} summaries.")
    if len(generated_summaries) > 0 and len(reference_summaries) > 0:
        logger.info(f"Example Generated Summary: {generated_summaries[0]}")
        logger.info(f"Example Reference Summary: {reference_summaries[0]}")

    # --- 5. Calculate ROUGE Scores ---
    if rouge_metric and len(generated_summaries) == len(reference_summaries) and len(generated_summaries) > 0:
        logger.info("Calculating ROUGE scores...")
        try:
            # Ensure summaries are strings before tokenization
            formatted_preds = ["\n".join(nltk.sent_tokenize(str(pred).strip())) for pred in generated_summaries]
            formatted_labels = ["\n".join(nltk.sent_tokenize(str(label).strip())) for label in reference_summaries]

            result = rouge_metric.compute(predictions=formatted_preds, references=formatted_labels, use_stemmer=True)
            result = {key: value * 100 for key, value in result.items()}

            logger.info("\n--- Evaluation Results ---")
            for key, value in result.items():
                logger.info(f"  {key}: {value:.4f}")
            logger.info("------------------------")

            # Save results and predictions
            output_eval_file = os.path.join(MODEL_PATH, "test_eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info(f"***** Writing results to {output_eval_file} *****")
                for key, value in result.items():
                    writer.write(f"{key} = {value:.4f}\n")

            output_preds_file = os.path.join(MODEL_PATH, "test_predictions.tsv")
            with open(output_preds_file, "w", encoding='utf-8') as writer:
                 writer.write("Generated Summary\tReference Summary\n")
                 for gen, ref in zip(generated_summaries, reference_summaries):
                      # Ensure strings before replacing tabs
                     gen_str = str(gen).replace(chr(9), ' ')
                     ref_str = str(ref).replace(chr(9), ' ')
                     writer.write(f"{gen_str}\t{ref_str}\n")
            logger.info(f"Predictions saved to {output_preds_file}")
        except Exception as e:
            logger.error(f"Error calculating or saving ROUGE scores: {e}", exc_info=True)

    elif not rouge_metric:
        logger.warning("ROUGE metric not loaded. Skipping score calculation.")
    elif len(generated_summaries) == 0:
         logger.warning("No summaries were generated. Skipping ROUGE calculation.")
    else:
        logger.warning(f"Mismatch between generated ({len(generated_summaries)}) and reference ({len(reference_summaries)}) summary counts. Skipping ROUGE calculation.")

    logger.info("Evaluation finished.")

if __name__ == "__main__":
    # *** Before running, double-check DOCUMENT_COLUMN and SUMMARY_COLUMN match your test.csv! ***
    logger.warning(f"Using DOCUMENT_COLUMN='{DOCUMENT_COLUMN}' and SUMMARY_COLUMN='{SUMMARY_COLUMN}'. Verify these match your CSV headers.")
    evaluate_summarizer()
