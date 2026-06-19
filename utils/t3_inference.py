"""
t3_inference.py — Inference module for HuggingFace Transformer models.

Provides:
  • BERT-based binary sentiment classification (SST-2)
  • GPT-2 text generation

Models are loaded once and cached via Streamlit's @st.cache_resource.
All tensor operations respect CUDA availability for transparent GPU acceleration.
"""

import time
import torch
import streamlit as st

# ---------------------------------------------------------------------------
# Helper constants — sample data for the Streamlit UI
# ---------------------------------------------------------------------------

SAMPLE_SENTIMENTS: list[str] = [
    "This movie was absolutely wonderful from start to finish.",
    "A terrible waste of time — the plot was incoherent and dull.",
    "The cinematography is breathtaking and the acting is superb.",
    "I found the storyline predictable and the characters flat.",
    "An outstanding film that keeps you on the edge of your seat.",
    "The dialogue felt forced and the pacing was painfully slow.",
    "A masterpiece of modern cinema with stellar performances.",
    "I was bored within the first ten minutes and it never improved.",
    "Visually stunning with a soundtrack that elevates every scene.",
    "One of the worst films I have ever had the displeasure of watching.",
]

SAMPLE_PROMPTS: dict[str, str] = {
    "Creative": "Once upon a time in a land far away,",
    "Scientific": "Recent advances in quantum computing have shown that",
    "Technical": "To implement a distributed microservices architecture,",
    "News": "Breaking news: Scientists announced today that",
    "Conversational": "Hey, I was thinking about what you said yesterday and",
}

# ---------------------------------------------------------------------------
# BERT — Sentiment Classification (SST-2)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_bert_model():
    """Load the pre-fine-tuned BERT model for SST-2 sentiment classification.

    Uses ``textattack/bert-base-uncased-SST-2`` which achieves >93 % accuracy
    on the Stanford Sentiment Treebank binary task.

    Returns
    -------
    tuple[AutoModelForSequenceClassification, AutoTokenizer, torch.device]
        (model, tokenizer, device) ready for inference.

    Raises
    ------
    RuntimeError
        If the model cannot be downloaded or loaded.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "textattack/bert-base-uncased-SST-2"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        return model, tokenizer, device

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load BERT sentiment model. Make sure the 'transformers' "
            f"library is installed and you have internet access for the initial "
            f"download.  Original error: {exc}"
        ) from exc


def predict_sentiment(
    model,
    tokenizer,
    device: torch.device,
    text: str,
) -> dict:
    """Run binary sentiment classification on a single text.

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        The loaded BERT model.
    tokenizer : AutoTokenizer
        The corresponding tokenizer.
    device : torch.device
        Target device (CPU / CUDA).
    text : str
        Input sentence or paragraph.

    Returns
    -------
    dict
        {
            'label':         'Positive' | 'Negative',
            'confidence':    float (0-1, probability of the predicted label),
            'positive_prob': float (0-1),
            'negative_prob': float (0-1),
        }
    """
    try:
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        probs = probabilities.squeeze().cpu().tolist()

        # SST-2 label mapping: 0 → Negative, 1 → Positive
        negative_prob = float(probs[0])
        positive_prob = float(probs[1])

        if positive_prob >= negative_prob:
            label = "Positive"
            confidence = positive_prob
        else:
            label = "Negative"
            confidence = negative_prob

        return {
            "label": label,
            "confidence": confidence,
            "positive_prob": positive_prob,
            "negative_prob": negative_prob,
        }

    except Exception as exc:
        return {
            "label": "Error",
            "confidence": 0.0,
            "positive_prob": 0.0,
            "negative_prob": 0.0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# GPT-2 — Text Generation
# ---------------------------------------------------------------------------


@st.cache_resource
def load_gpt2_model():
    """Load the GPT-2 language model for text generation.

    Returns
    -------
    tuple[GPT2LMHeadModel, GPT2Tokenizer, torch.device]
        (model, tokenizer, device) ready for generation.

    Raises
    ------
    RuntimeError
        If the model cannot be downloaded or loaded.
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model_name = "gpt2"

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        # GPT-2 has no native pad token; reuse EOS so batched generation works.
        tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        return model, tokenizer, device

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load GPT-2 model. Make sure the 'transformers' library "
            f"is installed and you have internet access for the initial download.  "
            f"Original error: {exc}"
        ) from exc


def generate_text(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.92,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> dict:
    """Generate text continuation from a prompt using GPT-2.

    Parameters
    ----------
    model : GPT2LMHeadModel
        The loaded GPT-2 model.
    tokenizer : GPT2Tokenizer
        The corresponding tokenizer.
    device : torch.device
        Target device (CPU / CUDA).
    prompt : str
        The text prompt to continue from.
    max_new_tokens : int
        Maximum number of *new* tokens to generate (excludes prompt tokens).
    temperature : float
        Sampling temperature — higher → more random.
    top_p : float
        Nucleus sampling cumulative probability threshold.
    top_k : int
        Top-k sampling parameter.
    repetition_penalty : float
        Penalty applied to already-generated tokens to reduce repetition.

    Returns
    -------
    dict
        {
            'text':          str  — full generated text (prompt + continuation),
            'time':          float — wall-clock generation time in seconds,
            'num_tokens':    int   — total tokens in the output sequence,
            'prompt_tokens': int   — number of tokens in the input prompt,
        }
    """
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_length = input_ids.shape[1]

        start_time = time.time()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - start_time

        generated_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return {
            "text": generated_text,
            "time": round(elapsed, 3),
            "num_tokens": int(output_ids.shape[1]),
            "prompt_tokens": int(prompt_length),
        }

    except Exception as exc:
        return {
            "text": f"[Generation failed: {exc}]",
            "time": 0.0,
            "num_tokens": 0,
            "prompt_tokens": 0,
            "error": str(exc),
        }
