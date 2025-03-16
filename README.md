# medIntel-
This repository contains MedIntel project related files

Frontend:
    Node.js
    Javascript
    HTML
    CSS

Backend:
Extracts text from forms (PDF, image, DOCX)
    ✅ Validates compliance against TennCare rules
    ✅ Flags missing/incorrect fields
    ✅ Provides suggestions for corrections

Required Libraried:
    Fine-tune a BERT model to detect missing fields and errors in text
    Train the model on sample form data
    fastapi → API backend
    uvicorn → Runs FastAPI
    pytesseract → Extracts text from images
    pdfplumber → Extracts text from PDFs
    docx → Extracts text from DOCX
    joblib → Loads trained AI models
    transformers → Pretrained BERT models
    torch → Deep learning backend (PyTorch)
    datasets → Loads datasets for fine-tuning
    scikit-learn → Evaluation metrics
    pandas → Handles structured data
    tqdm → Progress bar for training

Define TennCare Compliance Rules
    We will check for:
    ✅ Missing required fields
    ✅ Incorrect dates (age, waiting periods, expiration)
    ✅ Incorrect signatures and formats

Key Features
    ✅ Supports PDFs, Images, and DOCX
    ✅ Checks for missing required fields
    ✅ Validates waiting periods (30-180 days)
    ✅ Ensures recipient is 21+
    ✅ Flags invalid corrections (e.g., white-out use)
