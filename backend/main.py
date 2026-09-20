from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SatQuery AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "SatQuery AI Backend is running successfully!"
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    return {
        "filename": image.filename,
        "question": question,
        "answer": "Image received successfully. AI vision analysis will be connected next."
    }