import gradio as gr
from mask import mask_word_prediction


def greet(text):
    result = mask_word_prediction(text, 1)
    return result[0]


demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(
        label="Enter a sentence with a word replaced by an underscore (_) and the app will predict possible words to complete the sentence."
    ),
    outputs="text",
)
demo.launch()
