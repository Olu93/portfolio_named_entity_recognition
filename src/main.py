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
    models_dir = pathlib.Path(__file__).parent.parent / "models"
    
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
        with open(model_path, 'rb') as f:
            model = cloudpickle.load(f)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise



@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NER Entity Extraction API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
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
        required_columns = ['text', 'themes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}. Expected: {required_columns}"
            )
        
        # Initialize output columns
        df['persons'] = ''
        df['organizations'] = ''
        df['locations'] = ''
        
        # Bulk process the dataframe
        logger.info("Processing dataframe in bulk...")
        try:
            # Get all texts for bulk processing
            texts = df['text'].fillna('').astype(str).tolist()
            
            # Bulk predict
            predictions = model.predict(texts)
            
            # Process predictions in bulk
            for idx, prediction in enumerate(predictions):
                try:
                    # Extract entities based on your model's output format
                    if hasattr(prediction, 'persons'):
                        df.at[idx, 'persons'] = '; '.join(prediction.persons) if prediction.persons else ''
                    if hasattr(prediction, 'organizations'):
                        df.at[idx, 'organizations'] = '; '.join(prediction.organizations) if prediction.organizations else ''
                    if hasattr(prediction, 'locations'):
                        df.at[idx, 'locations'] = '; '.join(prediction.locations) if prediction.locations else ''
                    
                    # Alternative: if model returns a dictionary
                    elif isinstance(prediction, dict):
                        df.at[idx, 'persons'] = '; '.join(prediction.get('persons', []))
                        df.at[idx, 'organizations'] = '; '.join(prediction.get('organizations', []))
                        df.at[idx, 'locations'] = '; '.join(prediction.get('locations', []))
                    
                    # Alternative: if model returns a list of entity objects
                    elif isinstance(prediction, list):
                        persons = []
                        organizations = []
                        locations = []
                        
                        for entity in prediction:
                            if hasattr(entity, 'type') and hasattr(entity, 'value'):
                                if entity.type == 'PER':
                                    persons.append(entity.value)
                                elif entity.type == 'ORG':
                                    organizations.append(entity.value)
                                elif entity.type == 'LOC':
                                    locations.append(entity.value)
                        
                        df.at[idx, 'persons'] = '; '.join(persons)
                        df.at[idx, 'organizations'] = '; '.join(organizations)
                        df.at[idx, 'locations'] = '; '.join(locations)
                        
                except Exception as e:
                    logger.warning(f"Error processing prediction {idx}: {e}")
                    # Keep empty strings for failed predictions
                    continue
                    
        except Exception as e:
            logger.error(f"Error in bulk processing: {e}")
            raise
        
        # Create output file
        output_columns = ['persons', 'organizations', 'locations']
        output_df = df[output_columns]
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            output_df.to_csv(tmp_file.name, index=False)
            output_file_path = tmp_file.name
        
        logger.info(f"Processed {len(df)} rows successfully")
        
        return PredictionResponse(
            message="Processing completed successfully",
            output_file_path=output_file_path,
            rows_processed=len(df)
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

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
