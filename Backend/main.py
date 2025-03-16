import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
from fastapi import FastAPI, File, UploadFile
import pytesseract
from PIL import Image
import pdfplumber
import docx
import io
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

# Load trained BERT model & tokenizer
model = BertForSequenceClassification.from_pretrained("bert_model")
tokenizer = BertTokenizer.from_pretrained("bert_model")
label_mapping = joblib.load("label_mapping.pkl")
reverse_labels = {v: k for k, v in label_mapping.items()}

# Text extraction functions
def extract_text_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(pdf_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def extract_text_from_docx(docx_bytes):
    doc = docx.Document(io.BytesIO(docx_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def predict_errors(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return reverse_labels[predicted_label]

@app.post("/upload/")
async def upload_form(file: UploadFile = File(...)):
    contents = await file.read()
    file_type = file.filename.split(".")[-1].lower()
    
    if file_type in ["jpg", "jpeg", "png"]:
        text = extract_text_from_image(contents)
    elif file_type == "pdf":
        text = extract_text_from_pdf(contents)
    elif file_type == "docx":
        text = extract_text_from_docx(contents)
    else:
        return {"error": "Unsupported file format"}

    # Use BERT to predict errors
    error_prediction = predict_errors(text)

    return {
        "text": text[:500],
        "errors": error_prediction if error_prediction != "No errors" else None
    }
