import string
import tensorflow as tf
from transformers import AutoTokenizer, TFBertForMaskedLM

MODEL = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = TFBertForMaskedLM.from_pretrained(MODEL)


def mask_word_prediction(text_with_, K=1):
    results = []
    text = text_with_.replace("_", tokenizer.mask_token)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        print(f"Input must include mask token {tokenizer.mask_token}.")
        return

    mask_token_logits = model(**inputs).logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    filtered_tokens = [
        token
        for token in top_tokens
        if tokenizer.decode([token]) not in string.punctuation
    ]
    for token in filtered_tokens:
        result = text.replace(tokenizer.mask_token, tokenizer.decode([token]))
        results.append(result)

    return results


def get_mask_token_index(mask_token_id, inputs):
    token_ids = inputs["input_ids"][0].numpy()
    return list(token_ids).index(mask_token_id) if mask_token_id in token_ids else None
