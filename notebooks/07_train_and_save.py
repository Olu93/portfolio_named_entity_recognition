# %%
"""
Notebook to instantiate and cloud pickle the PretrainedBERTEntityExtractor model
"""

from notebook_config import FILES_DIR, MODEL_CONFIGS, WORK_DIR, MODELS_DIR
import os
import cloudpickle
import logging
from datetime import datetime
from port.entity_extractor import MultiEntityExtractor
from dotenv import load_dotenv, find_dotenv
import pandas as pd
# from src.constants.model_config import MODEL_CONFIGS
_ = load_dotenv(find_dotenv())




# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
def find_model_config(model_name: str, model_variant:str):
    for config in MODEL_CONFIGS:
        if config["name"] == model_name and model_variant == config["extra_info"]["model"]:
            return config
    raise ValueError(f"Model config not found for {model_name} and {model_variant}")

def create_and_save_model():
    """
    Create and save a PretrainedBERTEntityExtractor model for a specific entity type
    
    Args:
        entity_type: Type of entity to extract ('persons', 'organizations', 'locations')
        model_name: Optional custom name for the model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    model.fit(X, y)  # Empty list since we're using a pretrained model

    
    # Save the model using cloudpickle
    model_folder = MODELS_DIR / timestamp
    model_folder.mkdir(parents=True, exist_ok=True)

    model_path = model_folder / f"main_model.pkl"
    logger.info(f"Saving model to: {model_path}")
    
    with open(model_path, 'wb') as f:
        cloudpickle.dump(model, f)
    
    logger.info(f"Model saved successfully to {model_path}")
    
    return model, model_path


# %%
create_and_save_model()
# %%
