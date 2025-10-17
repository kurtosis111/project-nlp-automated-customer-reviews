import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# Load model and tokenizer from Hugging Face Hub
model_name = "yang181614/customer-review-distilbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Map label indices to sentiment
label_map = {0: "negative", 1: "neutral", 2: "positive"}

def classify_review(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    return label_map[pred]

# Gradio UI
gr.Interface(
    fn=classify_review,
    inputs=gr.Textbox(lines=4, placeholder="Paste your review here..."),
    outputs="label",
    title="Review Classifier",
    description="Upload a customer review and get its sentiment grade: negative, neutral, or positive."
).launch()
