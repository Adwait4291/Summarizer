import streamlit as st
import torch
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration # Explicit import for BART
)
from peft import PeftModel, PeftConfig
import logging
import time # To simulate loading time if needed

# --- Configuration ---
# *** Must match the OUTPUT_DIR from your training script ***
MODEL_PATH = "../saved_model/fine_tuned_summarizer_bart_custom"
# *** Must match the BASE_MODEL_NAME from your training script ***
BASE_MODEL_NAME = "facebook/bart-large-cnn"

# Generation parameters (can be adjusted)
MAX_SUMMARY_LENGTH = 150 # Max length for generated summaries in the app
MIN_SUMMARY_LENGTH = 30  # Min length
NUM_BEAMS = 5           # Beam search for potentially better results
LENGTH_PENALTY = 2.0    # Penalizes longer sequences more
NO_REPEAT_NGRAM_SIZE = 3 # Avoid repeating trigrams

# --- Setup Logging ---
# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model and Tokenizer Loading (Cached) ---
# Use Streamlit's caching to load the model only once
@st.cache_resource # Use cache_resource for non-data objects like models
def load_model_and_tokenizer(model_path, base_model_name):
    """Loads the fine-tuned model and tokenizer."""
    logger.info(f"Attempting to load model and tokenizer from: {model_path}")
    try:
        # Check if the path exists
        if not os.path.isdir(model_path):
             st.error(f"Model directory not found: {model_path}")
             st.error("Please ensure the fine-tuned model has been saved correctly after training.")
             return None, None

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer loaded successfully.")

        # Check if it's a PEFT/LoRA model
        is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if is_peft_model:
            logger.info("Detected PEFT/LoRA model. Loading base model and attaching adapter...")
            config = PeftConfig.from_pretrained(model_path)
            base_model_name_or_path = config.base_model_name_or_path
            if not base_model_name_or_path:
                 logger.warning(f"PEFT config does not specify base model. Using default: {base_model_name}")
                 base_model_name_or_path = base_model_name
            elif base_model_name_or_path != base_model_name:
                 logger.warning(f"PEFT config base model '{base_model_name_or_path}' differs from specified '{base_model_name}'. Using config's value.")

            # Load base model explicitly specifying the class if needed
            logger.info(f"Loading base model: {base_model_name_or_path}")
            # Use BartForConditionalGeneration for BART
            base_model = BartForConditionalGeneration.from_pretrained(base_model_name_or_path)
            logger.info("Loading PEFT adapter...")
            model = PeftModel.from_pretrained(base_model, model_path)
            # Optional: Merge adapter for faster inference (increases memory usage)
            # logger.info("Merging PEFT adapter...")
            # model = model.merge_and_unload()
            # logger.info("Adapter merged.")
            logger.info("PEFT model loaded successfully.")
        else:
            logger.info("Detected full fine-tuned model. Loading directly...")
            # Load the full fine-tuned model
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            logger.info("Full model loaded successfully.")

        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        logger.info(f"Model moved to {device} and set to evaluation mode.")

        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        logger.error(f"Error loading model/tokenizer from {model_path}: {e}", exc_info=True)
        return None, None

# --- Summarization Function ---
def summarize_text(model, tokenizer, text_to_summarize):
    """Generates a summary for the given text."""
    if not text_to_summarize or not text_to_summarize.strip():
        return "Please enter some text to summarize."

    try:
        device = model.device # Get device model is on
        logger.info(f"Tokenizing input text (max length: {tokenizer.model_max_length})...")
        # Tokenize input text - BART doesn't need a prefix
        inputs = tokenizer(
            text_to_summarize,
            max_length=tokenizer.model_max_length, # Use tokenizer's max length
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device) # Move tensors to the same device as the model

        logger.info("Generating summary...")
        # Generate summary
        with torch.no_grad(): # Ensure no gradients are calculated during inference
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_SUMMARY_LENGTH,
                min_length=MIN_SUMMARY_LENGTH,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=True # Stop when beams finish
            )

        logger.info("Decoding summary...")
        # Decode the generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info("Summary generation complete.")
        return summary

    except Exception as e:
        st.error(f"Error during summarization: {e}")
        logger.error(f"Error during summarization: {e}", exc_info=True)
        return "An error occurred while generating the summary."


# --- Streamlit App UI ---
st.set_page_config(layout="wide") # Use wide layout

st.title("📝 Text Summarizer App")
st.markdown("Paste your text below and click 'Summarize' to get a concise summary generated by a fine-tuned BART model.")

# Load model and tokenizer using caching
with st.spinner("Loading fine-tuned summarization model... Please wait."):
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, BASE_MODEL_NAME)

if model is None or tokenizer is None:
    st.error("Model loading failed. Please check the logs and ensure the model path is correct and files exist.")
    st.stop() # Stop execution if model loading fails
else:
    st.success("Model loaded successfully!")

    # Text area for user input
    input_text = st.text_area("Enter Text to Summarize:", height=250, placeholder="Paste your long text here...")

    # Button to trigger summarization
    if st.button("Summarize", type="primary"):
        if input_text and input_text.strip():
            with st.spinner("Generating summary..."):
                start_time = time.time()
                generated_summary = summarize_text(model, tokenizer, input_text)
                end_time = time.time()
                processing_time = end_time - start_time
                logger.info(f"Summarization took {processing_time:.2f} seconds.")

            st.subheader("Generated Summary:")
            st.markdown(f"> {generated_summary}") # Display summary in a blockquote
            st.info(f"Summarization completed in {processing_time:.2f} seconds.")
        else:
            st.warning("Please enter some text in the text area above.")

st.markdown("---")
st.markdown("Powered by Hugging Face Transformers & Streamlit")

