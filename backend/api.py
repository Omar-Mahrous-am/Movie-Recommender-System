from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import pandas as pd
import zipfile
from pathlib import Path
from typing import Optional

# Setup paths relative to the project root
# BASE_DIR is .../Movie-Recommender-System
BASE_DIR = Path(__file__).resolve().parent.parent
PKL_FILE = BASE_DIR / "cosine_sim.pkl"
ZIP_FILE = BASE_DIR / "cosine_sim.zip"
CSV_FILE = BASE_DIR / "movies.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

# Unzip cosine_sim.pkl only if it doesn't exist
if not PKL_FILE.exists():
    if ZIP_FILE.exists():
        print(f"Extracting {ZIP_FILE}...")
        try:
            with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
                zip_ref.extractall(BASE_DIR)
        except Exception as e:
            print(f"Error extracting zip: {e}")
    else:
        print(f"Warning: {ZIP_FILE} not found. Ensure movie data is available.")

# Load data into memory
try:
    if PKL_FILE.exists():
        with open(PKL_FILE, "rb") as f:
            cosine_sim = pickle.load(f)
    else:
        cosine_sim = None
        print(f"Warning: {PKL_FILE} not found.")

    if CSV_FILE.exists():
        df2 = pd.read_csv(CSV_FILE)
        indices = pd.Series(df2.index, index=df2['title']).to_dict()
    else:
        df2 = None
        indices = {}
        print(f"Warning: {CSV_FILE} not found.")
except Exception as e:
    print(f"Error loading model data: {e}")
    cosine_sim = None
    df2 = None
    indices = {}

app = FastAPI(title="Movie Recommender API")

# Enable CORS for frontend communication
# This allows your index.html to talk to this API even if opened as a file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    title: str

class QuizRequest(BaseModel):
    title: str
    genre: Optional[str] = None
    era: Optional[str] = None

@app.post("/predict")
def predict(request: PredictionRequest):
    if cosine_sim is None or df2 is None or not indices:
        raise HTTPException(status_code=500, detail="Model data could not be loaded or files are missing.")
        
    title = request.title
    if title not in indices:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found in dataset")
        
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df2['title'].iloc[movie_indices].tolist()
    
    return {
        "requested_movie": title,
        "recommendations": recommendations
    }

@app.post("/quiz-recommend")
def quiz_recommend(request: QuizRequest):
    """
    Endpoint for the frontend quiz feature. 
    Currently aliases to the standard prediction logic.
    """
    return predict(PredictionRequest(title=request.title))

# Mount the frontend directory to serve index.html at the root "/"
# IMPORTANT: This must be defined AFTER the API routes
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"message": "Welcome to the Movie Recommender System API (Frontend not found)"}
