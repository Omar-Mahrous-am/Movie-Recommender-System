from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import pandas as pd
import zipfile
import logging
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("movie-recommender")

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
# Primary: project root derived from this file's location (backend/api.py)
_FILE_DIR = Path(__file__).resolve().parent          # .../backend/
_PROJECT_ROOT = _FILE_DIR.parent                      # .../Movie-Recommender-System/ (or /app in Docker)

# Secondary: the current working directory (handles `uvicorn backend.api:app` from /app)
_CWD = Path.cwd()


def _find_file(filename: str) -> Optional[Path]:
    """Search for a file across multiple candidate directories.
    
    Returns the first existing path, or None.
    Candidate order:
      1. Project root (derived from __file__)
      2. Current working directory
      3. data/ subdirectories of both
    """
    candidates = [
        _PROJECT_ROOT / filename,
        _CWD / filename,
        _PROJECT_ROOT / "data" / filename,
        _CWD / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            logger.info(f"Found '{filename}' at {path}")
            return path
    logger.warning(f"'{filename}' not found in any candidate path: {[str(c) for c in candidates]}")
    return None


def _find_directory(dirname: str) -> Optional[Path]:
    """Search for a directory across project root and CWD."""
    candidates = [
        _PROJECT_ROOT / dirname,
        _CWD / dirname,
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None


# ---------------------------------------------------------------------------
# Zip Extraction
# ---------------------------------------------------------------------------
def _extract_pkl_from_zip() -> None:
    """Extract cosine_sim.pkl from cosine_sim.zip if the pkl doesn't exist."""
    pkl_path = _find_file("cosine_sim.pkl")
    if pkl_path is not None:
        logger.info("cosine_sim.pkl already exists — skipping extraction.")
        return

    zip_path = _find_file("cosine_sim.zip")
    if zip_path is None:
        logger.warning("cosine_sim.zip not found — cannot extract model file.")
        return

    # Extract into the same directory that contains the zip
    extract_dir = zip_path.parent
    logger.info(f"Extracting {zip_path} → {extract_dir} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        logger.error(f"Failed to extract zip: {e}")
        return

    # Verify the expected file was actually produced
    expected = extract_dir / "cosine_sim.pkl"
    if expected.exists():
        logger.info(f"Successfully extracted {expected} ({expected.stat().st_size:,} bytes)")
    else:
        logger.error(
            f"Extraction completed but cosine_sim.pkl not found at {expected}. "
            f"Zip contents: {[f.filename for f in zipfile.ZipFile(zip_path).filelist]}"
        )


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
# Global model state — None means "not loaded"
_cosine_sim = None
_df = None
_indices: dict = {}
_model_loaded = False
_load_error: Optional[str] = None


def load_model() -> None:
    """Load the similarity matrix and movie dataset into memory.
    
    Designed to be called once at startup.  Never raises — stores
    error state internally so the API can report it at request time.
    """
    global _cosine_sim, _df, _indices, _model_loaded, _load_error

    # Step 1: ensure pkl exists (extract from zip if needed)
    _extract_pkl_from_zip()

    # Step 2: load cosine similarity matrix
    pkl_path = _find_file("cosine_sim.pkl")
    if pkl_path is None:
        _load_error = "cosine_sim.pkl could not be found after extraction attempt."
        logger.error(_load_error)
        return

    try:
        with open(pkl_path, "rb") as f:
            _cosine_sim = pickle.load(f)
        logger.info(f"Loaded cosine_sim matrix from {pkl_path}")
    except Exception as e:
        _load_error = f"Failed to load cosine_sim.pkl: {e}"
        logger.error(_load_error)
        return

    # Step 3: load movies dataset
    csv_path = _find_file("movies.csv")
    if csv_path is None:
        _load_error = "movies.csv could not be found."
        logger.error(_load_error)
        return

    try:
        _df = pd.read_csv(csv_path)
        _indices = pd.Series(_df.index, index=_df["title"]).to_dict()
        logger.info(f"Loaded {len(_df):,} movies from {csv_path}")
    except Exception as e:
        _load_error = f"Failed to load movies.csv: {e}"
        logger.error(_load_error)
        return

    _model_loaded = True
    _load_error = None
    logger.info("✓ Model loaded successfully — API is ready for predictions.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Movie Recommender API")


@app.on_event("startup")
def startup_event():
    """Load model data when the FastAPI application starts."""
    load_model()

# CORS — allow the frontend (served from file:// or a different origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    title: str


class QuizRequest(BaseModel):
    title: str
    genre: Optional[str] = None
    era: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _ensure_model_ready() -> None:
    """Raise HTTP 500 if the model is not available."""
    if not _model_loaded:
        detail = _load_error or "Model data could not be loaded or files are missing."
        raise HTTPException(status_code=500, detail=detail)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Lightweight health-check endpoint for container orchestrators."""
    return {
        "status": "healthy" if _model_loaded else "degraded",
        "model_loaded": _model_loaded,
        "error": _load_error,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    _ensure_model_ready()

    title = request.title
    if title not in _indices:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found in dataset")

    idx = _indices[title]
    sim_scores = list(enumerate(_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 recommendations

    movie_indices = [i[0] for i in sim_scores]
    recommendations = _df["title"].iloc[movie_indices].tolist()

    return {
        "requested_movie": title,
        "recommendations": recommendations,
    }


@app.post("/quiz-recommend")
def quiz_recommend(request: QuizRequest):
    """Endpoint for the frontend quiz feature.
    Currently aliases to the standard prediction logic.
    """
    return predict(PredictionRequest(title=request.title))


# ---------------------------------------------------------------------------
# Static Frontend
# ---------------------------------------------------------------------------
# Mount AFTER API routes so /predict and /quiz-recommend take priority
_frontend_dir = _find_directory("frontend")
if _frontend_dir is not None:
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    logger.info(f"Serving frontend from {_frontend_dir}")
else:
    logger.warning("Frontend directory not found — serving API-only mode.")

    @app.get("/")
    def read_root():
        return {"message": "Welcome to the Movie Recommender System API (frontend not found)"}
