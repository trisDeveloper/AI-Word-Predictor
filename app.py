import gradio as gr
from mask import mask_word_prediction


def greet(text):
    result = mask_word_prediction(text, 1)
    return result[0]


demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Enter a sentence with a masked word ([MASK]):"),
    outputs="text",
)
demo.launch()
