import os
import sys
import pathlib
import cloudpickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import logging
from contextlib import asynccontextmanager
import pickle

from constants.values import MODELS_DIR
from utils.misc import load_model_cross_platform

# Add the project root to the path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global variable to store the loaded model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    yield
    # Shutdown
    pass

app = FastAPI(
    title="NER Entity Extraction API",
    description="API for extracting persons, organizations, and locations from text",
    version="1.0.0",
    lifespan=lifespan
)


class PredictionResponse(BaseModel):
    message: str
    output_file_path: str
    rows_processed: int

def load_model():
    """Load the model from cloudpickle file"""
    global model
    
    # Look for model files in the models directory
    models_dir = MODELS_DIR
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Find the most recent model folder
    model_folders = [f for f in models_dir.iterdir() if f.is_dir()]
    if not model_folders:
        raise FileNotFoundError(f"No model folders found in {models_dir}")
    
    # Sort by modification time and get the most recent
    latest_model_folder = max(model_folders, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading model from: {latest_model_folder}")
    
    # Look for cloudpickle files
    pickle_files = list(latest_model_folder.glob("*.pkl"))
    if not pickle_files:
        raise FileNotFoundError(f"No .pkl files found in {latest_model_folder}")
    
    # Load the first pickle file found
    model_path = pickle_files[0]
    logger.info(f"Loading model from: {model_path}")
    
    try:
        model = load_model_cross_platform(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise



@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "NER Entity Extraction API", "status": "running"}

@app.get("/healthz")
async def health_check():
    """Health check endpoint for load testing and monitoring"""
    try:
        # Check if model is loaded
        model_status = "loaded" if model is not None else "not_loaded"
        
        # Try to load model if not loaded
        if model is None:
            try:
                load_model()
                model_status = "loaded"
            except Exception as e:
                model_status = f"failed_to_load: {str(e)}"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "service": "NER Entity Extraction API"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "NER Entity Extraction API"
        }

@app.post("/predict/bulk", response_model=PredictionResponse)
async def predict_entities(file: UploadFile = File(...)):
    """
    Extract entities from CSV file containing text data.
    
    Input CSV should have columns: ['text', 'themes']
    Output CSV will have columns: ['persons', 'organizations', 'locations']
    """
    
    # Check if model is loaded
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not available: {str(e)}")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file.file)
        logger.info(f"Loaded CSV with {len(df)} rows")
        
        # Validate required columns
        
        if 'text' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: text"
            )
        
        # Initialize output columns
        df['persons'] = ''
        df['organizations'] = ''
        df['locations'] = ''
        
        # Bulk process the dataframe
        logger.info("Processing dataframe in bulk...")
        # Get all texts for bulk processing
        texts = df['text'].fillna('').astype(str).tolist()
        
        # Bulk predict
        predictions: pd.DataFrame = model.predict(texts)

        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            predictions.to_csv(tmp_file.name, index=False)
            output_file_path = tmp_file.name
        
        logger.info(f"Processed {len(predictions)} rows successfully")
        
        return PredictionResponse(
            message="Processing completed successfully",
            output_file_path=output_file_path,
            rows_processed=len(df)
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict/single")
async def predict_single_text(request: dict):
    """
    Extract entities from a single text string (JSON endpoint for load testing).
    
    Input: {"text": "your text here"}
    Output: {"persons": [...], "organizations": [...], "locations": [...]}
    """
    # Check if model is loaded
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not available: {str(e)}")
    
    # Validate input
    if "text" not in request:
        raise HTTPException(status_code=400, detail="Missing 'text' field in request")
    
    text = request["text"]
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Text must be a non-empty string")
    
    try:
        # Predict entities for the single text
        predictions = model.predict([text])
        
        # Extract the first (and only) prediction
        result = {
            "persons": predictions.iloc[0]["persons"] if len(predictions) > 0 else [],
            "organizations": predictions.iloc[0]["organizations"] if len(predictions) > 0 else [],
            "locations": predictions.iloc[0]["locations"] if len(predictions) > 0 else []
        }
        
        logger.info(f"Processed single text successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Download the processed CSV file"""
    try:
        # Validate file path to prevent directory traversal
        if '..' in file_path or file_path.startswith('/'):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename="predictions.csv",
            media_type="text/csv"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"message": "No model loaded"}
    
    return {
        "model_type": type(model).__name__,
        "model_attributes": [attr for attr in dir(model) if not attr.startswith('_')],
        "has_predict": hasattr(model, 'predict'),
        "has_predict_proba": hasattr(model, 'predict_proba')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
