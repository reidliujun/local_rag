import gradio as gr
from query import rag_query

def gradio_query(question):
    return rag_query(question)

iface = gr.Interface(
    fn=gradio_query,
    inputs="text",
    outputs="text",
    title="Local RAG"
)
iface.launch()