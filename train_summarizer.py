import os
import torch
import numpy as np
from datasets import load_dataset, DatasetDict, Features, Value
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
# Removed PEFT imports
# Removed evaluate and nltk imports (as ROUGE is removed)
import logging
from functools import partial # To pass tokenizer to map function
import sys # For exiting script on critical errors

# --- Configuration ---
# Model Selection
MODEL_NAME = "t5-small"

# Paths (relative to the script execution location)
DATA_DIR = "./data"                   # Directory containing train.csv, validation.csv
OUTPUT_DIR = "./saved_model/fine_tuned_summarizer_t5_simple" # Directory to save the fine-tuned model
LOG_DIR = "./logs_summarizer_t5_simple"      # Directory for training logs

# Dataset Column Names (CRITICAL: Must match your CSV headers)
DOCUMENT_COLUMN = "dialogue" # Or 'text', 'article', etc.
SUMMARY_COLUMN = "summary"   # Or 'headlines', etc.

# T5 Specific Prefix (Required for summarization task with T5 models)
PREFIX = "summarize: "

# Tokenization parameters
MAX_INPUT_LENGTH = 512 # T5-small's standard max length
MAX_TARGET_LENGTH = 128 # Max length for generated summaries

# LoRA/PEFT Configuration - DISABLED

# Training Hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 8 # Per device batch size (adjust based on GPU VRAM)
GRADIENT_ACCUMULATION_STEPS = 2 # Simulate larger batch size
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine" # Learning rate scheduler type
LOGGING_STEPS = 50           # Log metrics (loss) every N steps
EVAL_STEPS = 200             # Evaluate loss on validation set every N steps
SAVE_STEPS = 200             # Save a checkpoint every N steps
EARLY_STOPPING_PATIENCE = 3  # Stop training if validation loss doesn't improve

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # logging.FileHandler(os.path.join(LOG_DIR, "training_simple.log"))
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging setup complete.")
logger.info(f"Script arguments: MODEL_NAME={MODEL_NAME}, DATA_DIR={DATA_DIR}, OUTPUT_DIR={OUTPUT_DIR}")
logger.info(f"Columns: DOCUMENT='{DOCUMENT_COLUMN}', SUMMARY='{SUMMARY_COLUMN}'")
logger.info(f"Training Params: EPOCHS={NUM_EPOCHS}, BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM={GRADIENT_ACCUMULATION_STEPS}, LR={LEARNING_RATE}")
logger.info("LoRA/PEFT Disabled. Performing full fine-tuning.")

# --- ROUGE Metric Loading Removed ---

# --- Preprocessing Function ---
def preprocess_function(examples, tokenizer):
    """Tokenizes documents and summaries for T5, adding the required prefix."""
    inputs = examples[DOCUMENT_COLUMN]
    targets = examples[SUMMARY_COLUMN]
    if not isinstance(inputs, list): inputs = [inputs]
    if not isinstance(targets, list): targets = [targets]

    inputs = [PREFIX + ("" if text is None else str(text)) for text in inputs]
    targets = ["" if text is None else str(text) for text in targets]

    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Compute Metrics Function Removed ---

# --- Main Training Function ---
def train():
    """Main function to handle data loading, preprocessing, training, and saving."""
    logger.info("Starting simplified summarization fine-tuning process with T5-Small...")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # === 1. Load Dataset ===
    logger.info(f"Loading datasets from CSV files in directory: {DATA_DIR}")
    train_file = os.path.join(DATA_DIR, 'train.csv')
    val_file = os.path.join(DATA_DIR, 'validation.csv')

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        logger.error(f"Training ({train_file}) or Validation ({val_file}) file not found.")
        logger.error(f"Please ensure train.csv and validation.csv are in the '{DATA_DIR}' directory.")
        sys.exit(1)

    try:
        dataset = load_dataset('csv', data_files={'train': train_file, 'validation': val_file})
        logger.info(f"Dataset loaded successfully: {dataset}")

        logger.info("Verifying dataset column names...")
        train_columns = dataset['train'].column_names
        if DOCUMENT_COLUMN not in train_columns:
             logger.error(f"Document column '{DOCUMENT_COLUMN}' not found in {train_file}. Found columns: {train_columns}.")
             sys.exit(1)
        if SUMMARY_COLUMN not in train_columns:
             logger.error(f"Summary column '{SUMMARY_COLUMN}' not found in {train_file}. Found columns: {train_columns}.")
             sys.exit(1)
        logger.info("Dataset column names verified.")

    except Exception as e:
        logger.error(f"Fatal error loading or verifying dataset from {DATA_DIR}: {e}", exc_info=True)
        sys.exit(1)

    # === 2. Load Tokenizer ===
    logger.info(f"Loading tokenizer for model: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading tokenizer '{MODEL_NAME}': {e}", exc_info=True)
        sys.exit(1)

    # === 3. Preprocess Data ===
    logger.info(f"Preprocessing datasets (Max input: {MAX_INPUT_LENGTH}, Max target: {MAX_TARGET_LENGTH})...")
    preprocess_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    try:
        tokenized_datasets = dataset.map(
            preprocess_with_tokenizer,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        logger.info("Dataset preprocessing complete.")
    except Exception as e:
        logger.error(f"Fatal error during dataset mapping/preprocessing: {e}", exc_info=True)
        sys.exit(1)

    # === 4. Load Model ===
    logger.info(f"Loading base model: {MODEL_NAME}")
    try:
        # Performing full fine-tuning, so load the base model directly
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        # Removed the line causing the error: T5Config has no 'max_position_embeddings'
        logger.info(f"Base model '{MODEL_NAME}' loaded.")
        # General warning about input length
        logger.warning(f"Ensure MAX_INPUT_LENGTH ({MAX_INPUT_LENGTH}) is appropriate for the {MODEL_NAME} model. Input will be truncated if longer.")
    except Exception as e:
        logger.error(f"Fatal error loading base model '{MODEL_NAME}': {e}", exc_info=True)
        sys.exit(1)

    # === 5. LoRA/PEFT Application Removed ===
    logger.info("Performing full fine-tuning (LoRA/PEFT is disabled).")

    # === 6. Define Training Arguments ===
    logger.info("Defining Seq2Seq training arguments...")
    try:
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            logging_dir=LOG_DIR,
            logging_strategy="steps",
            logging_steps=LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=2,
            load_best_model_at_end=True, # Load best model based on validation loss
            metric_for_best_model="loss", # Use loss to determine best model
            greater_is_better=False,     # Lower loss is better
            # predict_with_generate=False, # Not needed without ROUGE/generation metrics
            report_to="tensorboard",
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
        )
        logger.info("Training arguments defined.")
    except Exception as e:
        logger.error(f"Fatal error defining TrainingArguments: {e}", exc_info=True)
        sys.exit(1)

    # === 7. Initialize Data Collator ===
    logger.info("Initializing Data Collator...")
    try:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        logger.info("Data Collator initialized.")
    except Exception as e:
         logger.error(f"Fatal error initializing DataCollator: {e}", exc_info=True)
         sys.exit(1)

    # === 8. Initialize Trainer ===
    logger.info("Initializing Seq2SeqTrainer...")
    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None, # No compute_metrics function passed
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
        )
        logger.info("Seq2SeqTrainer initialized.")
    except Exception as e:
        logger.error(f"Fatal error initializing Seq2SeqTrainer: {e}", exc_info=True)
        sys.exit(1)

    # === 9. Start Training ===
    logger.info("Starting model training...")
    train_result = None
    try:
        train_result = trainer.train()
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.error(f"Error occurred during training loop: {e}", exc_info=True)
        try:
            logger.warning("Attempting to save trainer state after training error...")
            trainer.save_state()
            logger.info("Trainer state saved.")
        except Exception as save_e:
            logger.error(f"Could not save trainer state after error: {save_e}", exc_info=True)
        sys.exit(1)

    # === 10. Save the Final Model & Tokenizer ===
    final_output_dir = training_args.output_dir
    logger.info(f"Saving the best model and tokenizer to {final_output_dir}...")
    try:
        os.makedirs(final_output_dir, exist_ok=True)
        # Save the full fine-tuned model (as LoRA is disabled)
        trainer.save_model(final_output_dir)
        logger.info("Full fine-tuned model saved.")
        # Always save the tokenizer
        tokenizer.save_pretrained(final_output_dir)
        logger.info("Tokenizer saved successfully.")
    except Exception as e:
        logger.error(f"Error saving final model/tokenizer: {e}", exc_info=True)

    # === 11. Log and Save Training Metrics ===
    if train_result:
        try:
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            logger.info(f"Training metrics saved: {metrics}")
        except Exception as e:
            logger.error(f"Error logging/saving training metrics: {e}", exc_info=True)
    else:
        logger.warning("Training did not complete successfully, skipping saving of final training metrics.")

    # === 12. Final Evaluation Removed (as ROUGE is not calculated) ===
    logger.info("Skipping final evaluation step as ROUGE metrics are disabled.")

    logger.info("Simplified fine-tuning script finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating directories: {e}")
        sys.exit(1)

    train_path = os.path.join(DATA_DIR, 'train.csv')
    val_path = os.path.join(DATA_DIR, 'validation.csv')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.error(f"Training ({train_path}) or Validation ({val_path}) file not found.")
        logger.error(f"Please ensure train.csv and validation.csv are in the '{DATA_DIR}' directory.")
        sys.exit(1)
    else:
        logger.info("Found train.csv and validation.csv.")
        logger.warning(f"Using DOCUMENT_COLUMN='{DOCUMENT_COLUMN}' and SUMMARY_COLUMN='{SUMMARY_COLUMN}'. Verify these match your CSV headers.")
        train()

