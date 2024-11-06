import string
import tensorflow as tf
from transformers import AutoTokenizer, TFBertForMaskedLM

MODEL = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = TFBertForMaskedLM.from_pretrained(MODEL)


def mask_word_prediction(text_with_, K=1):
    # Replace all "_" with the model's mask
    text = text_with_.replace("_", tokenizer.mask_token)
    inputs = tokenizer(text, return_tensors="tf")

    mask_token_indices = get_mask_token_indices(tokenizer.mask_token_id, inputs)
    if not mask_token_indices:
        print(f"Input must include mask token {tokenizer.mask_token}.")
        return []

    results = []
    for mask_token_index in mask_token_indices:
        inputs = tokenizer(text, return_tensors="tf")

        mask_token_logits = model(**inputs).logits[0, mask_token_index]
        top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()

        filtered_tokens = [
            token
            for token in top_tokens
            if tokenizer.decode([token]) not in string.punctuation
        ]

        if filtered_tokens:
            predicted_word = tokenizer.decode([filtered_tokens[0]])
            text = text.replace(tokenizer.mask_token, predicted_word, 1)

    results.append(text)
    return results


def get_mask_token_indices(mask_token_id, inputs):
    token_ids = inputs["input_ids"][0].numpy()
    return [i for i, token_id in enumerate(token_ids) if token_id == mask_token_id]
