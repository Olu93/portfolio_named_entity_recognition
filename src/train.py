#!/usr/bin/env python3
"""
Script to instantiate and save the PretrainedBERTEntityExtractor model
"""

import argparse
from constants.values import WORK_DIR, MODELS_DIR
from constants.model_config import MODEL_CONFIGS
import os
import logging
from datetime import datetime
from port.entity_extractor import MultiEntityExtractor
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import sys

# Add the project root to the path to import utils
# sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils.misc import save_model_cross_platform

# from src.constants.model_config import MODEL_CONFIGS
_ = load_dotenv(find_dotenv())

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_model_config(model_name: str, model_variant: str):
    """Find model configuration by name and variant."""
    for config in MODEL_CONFIGS:
        if config["name"] == model_name and model_variant == config["extra_info"]["model"]:
            return config
    raise ValueError(f"Model config not found for {model_name} and {model_variant}")


def create_and_save_model(prepare_serialization: bool = True):
    """
    Create and save a PretrainedBERTEntityExtractor model for all entity types
    
    Args:
        prepare_serialization: Whether to call _prepare_serialization() before saving
    
    Returns:
        tuple: (model, model_path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get model configurations from environment variables
    MODEL_NAME_PERSON = os.getenv("MODEL_NAME_PERSON")
    MODEL_VARIANT_PERSON = os.getenv("MODEL_VARIANT_PERSON")
    model_config_person = find_model_config(MODEL_NAME_PERSON, MODEL_VARIANT_PERSON)
    
    MODEL_NAME_ORGANISATION = os.getenv("MODEL_NAME_ORGANISATION")
    MODEL_VARIANT_ORGANISATION = os.getenv("MODEL_VARIANT_ORGANISATION")
    model_config_organization = find_model_config(MODEL_NAME_ORGANISATION, MODEL_VARIANT_ORGANISATION)
    
    MODEL_NAME_LOCATION = os.getenv("MODEL_NAME_LOCATION")
    MODEL_VARIANT_LOCATION = os.getenv("MODEL_VARIANT_LOCATION")
    model_config_location = find_model_config(MODEL_NAME_LOCATION, MODEL_VARIANT_LOCATION)
    
    # Instantiate the model
    model = MultiEntityExtractor()
    model.add_extractor("persons", model_config_person["persons"]["extractor"](**model_config_person["persons"]["params"]))
    model.add_extractor("organizations", model_config_organization["organizations"]["extractor"](**model_config_organization["organizations"]["params"]))
    model.add_extractor("locations", model_config_location["locations"]["extractor"](**model_config_location["locations"]["params"]))
    
    # Fit the model (this will load the pretrained weights)
    logger.info("Loading pretrained model...")

    train_df = pd.read_csv(WORK_DIR / os.getenv("TRAIN_DATA_PATH")).fillna("")
    X = train_df["text"]
    y = train_df[["persons", "organizations", "locations"]]
    model.fit(X, y)

    # Save the model using cross-platform cloudpickle
    model_folder = MODELS_DIR / timestamp
    model_folder.mkdir(parents=True, exist_ok=True)

    model_path = model_folder / f"main_model.pkl"
    logger.info(f"Saving model to: {model_path}")
    
    # Prepare serialization if requested
    if prepare_serialization:
        logger.info("Preparing model for serialization...")
        model._prepare_serialization()
    
    # Use cross-platform saving function
    save_model_cross_platform(model, model_path)
    
    logger.info(f"Model saved successfully to {model_path}")
    
    return model, model_path


def main():
    """Main function to handle command line arguments and execute training."""
    parser = argparse.ArgumentParser(description="Train and save a multi-entity extraction model")
    parser.add_argument(
        "--prepare-serialization", 
        action="store_true",
        help="Call _prepare_serialization() before saving the model"
    )
    
    args = parser.parse_args()
    
    # Convert the flag to the boolean parameter
    prepare_serialization = args.prepare_serialization
    
    try:
        model, model_path = create_and_save_model(prepare_serialization=prepare_serialization)
        logger.info(f"Training completed successfully. Model saved to: {model_path}")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
