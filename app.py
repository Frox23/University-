from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import cv2
import numpy as np
import pytesseract
import re
import os
from sentence_transformers import SentenceTransformer

from RAG import get_rag_answer, update_index_with_new_text

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


class QuestionRequest(BaseModel):
    question: str

@app.post("/combined")
async def combined(request: QuestionRequest):
    try:
        answer = get_rag_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Failed to process question: {str(e)}"}
        )

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        html_path = "static/index.html"
        if not os.path.exists(html_path):
            return HTMLResponse(content="""
            <html>
                <body>
                    <h1>Medicine OCR & RAG System</h1>
                    <p>Static files not found. Please create static/index.html</p>
                </body>
            </html>
            """)
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<html><body><h1>Error: {str(e)}</h1></body></html>")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400, 
                content={"error": "File must be an image"}
            )
        
        contents = await file.read()
        processed_img = preprocess_image(contents)

        raw_text = pytesseract.image_to_string(
            processed_img, 
            config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.:,;()- \n"
        ).strip()
        
        print("🧾 Extracted text:\n", raw_text)
        
        if not raw_text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text could be extracted from the image"}
            )

        append_raw_text_to_txt_file(raw_text)
        
        model_instance = get_model()
        update_success = update_index_with_new_text(raw_text, model_instance)
        
        parsed_data = parse_leaflet_text(raw_text)

        return JSONResponse(content={
            "raw_text": raw_text,
            "parsed_data": parsed_data,
            "index_updated": update_success
        })

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Failed to process image: {str(e)}"}
        )


def preprocess_image(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image data")
        
        height, width = img.shape[:2]
        if height < 1000 or width < 1000:
            scale_factor = max(1000/height, 1000/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        denoised = cv2.fastNlMeansDenoising(gray)
        
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    except Exception as e:
        print(f"❌ Image preprocessing error: {e}")
        raise

def append_raw_text_to_txt_file(text, filename="Medicines(1).txt"):
    try:
        with open(filename, mode="a", encoding="utf-8") as f:
            f.write(f"\n---\n{text}\n")
        print(f"✅ Appended raw text to: {filename}")
    except Exception as e:
        print(f"❌ File write error: {e}")

def parse_leaflet_text(text):
    text = re.sub(r"(?i)MEDICINE\s+LEAFLET", "", text)
    text = re.sub(r'\s+', ' ', text)  
    
    fields = {
        "MEDICINE NAME": "",
        "ACTIVE INGREDIENTS": "",
        "DOSAGE": "",
        "INDICATIONS": "",
        "SIDE EFFECTS": "",
        "WARNINGS & PRECAUTIONS": "",
        "CONTRAINDICATIONS": ""
    }
    field_variations = {
        "MEDICINE NAME": ["MEDICINE NAME", "NAME", "Drug Name"],
        "ACTIVE INGREDIENTS": ["ACTIVE INGREDIENTS", "ACTIVE INGREDIENT", "Composition"],
        "DOSAGE": ["DOSAGE", "DOSE", "Dosing"],
        "INDICATIONS": ["INDICATIONS", "INDICATION", "Uses"],
        "SIDE EFFECTS": ["SIDE EFFECTS", "Adverse Effects", "Side Effects"],
        "WARNINGS & PRECAUTIONS": ["WARNINGS & PRECAUTIONS", "WARNINGS", "PRECAUTIONS"],
        "CONTRAINDICATIONS": ["CONTRAINDICATIONS", "Contraindication"]
    }

    for standard_key, variations in field_variations.items():
        for variant in variations:
            pattern = rf"(?i)\b{re.escape(variant)}\s*:?\s*(.*?)(?=\n\s*(?:[A-Z][A-Z\s&]+?\s*:|$))"
            match = re.search(pattern, text, flags=re.DOTALL)
            if match and match.group(1).strip():
                fields[standard_key] = match.group(1).strip().replace("\n", " ")
                break

    print("📋 Parsed data:", fields)
    return fields

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Medicine OCR & RAG API is running"}
