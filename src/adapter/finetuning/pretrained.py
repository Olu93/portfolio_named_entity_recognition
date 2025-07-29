from port.entity_extractor import SingleEntityExtractor
from utils.develop import test_extractor
from utils.typings import TextInput
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langchain.text_splitter import TokenTextSplitter
import torch
import logging
import pathlib
import sys

# Add the project root to the path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

from constants.values import FILES_DIR

logger = logging.getLogger(__name__)


class PretrainedBERTEntityExtractor(SingleEntityExtractor):
    """
    Entity extractor using a fine-tuned BERT model for NER.
    """

    def __init__(
        self,
        model_path: str = None,
        label: str = "persons",
        require_full_name: bool = True,
        aggregation_strategy: str = "first",
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        *args,
        **kwargs
    ):
        super().__init__(label=label, *args, **kwargs)
        
        # Set default model path if not provided
        if model_path is None:
            model_path = str(FILES_DIR / "pretrained" / "bert_ner_finetuned")
        
        self.model_path = model_path
        self.require_full_name = require_full_name
        self.aggregation_strategy = aggregation_strategy
        
        # Initialize text splitter
        self.text_splitter = TokenTextSplitter(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            self.device = 0
        else:
            logger.info("CUDA is not available. Using CPU for inference.")
            self.device = -1

    def _load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            
            # Create NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy=self.aggregation_strategy,
                device=self.device
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _fit(self, X: TextInput, y: TextInput = None):
        """Load the model (no training needed for pretrained model)"""
        if self.ner_pipeline is None:
            self._load_model()
        return self

    def _predict(self, X: TextInput):
        """Extract entities from texts using the fine-tuned model"""
        if self.ner_pipeline is None:
            self._load_model()
        
        output = []
        
        for text in X:
            try:
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                # Extract entities from all chunks
                all_entities = []
                for chunk in chunks:
                    # Get NER predictions for this chunk
                    spans = self.ner_pipeline(chunk)
                    
                    # Extract entities based on label type
                    for span in spans:
                        entity_text = span["word"]
                        entity_group = span["entity_group"]
                        
                        # Filter by entity type based on label
                        if self._is_valid_entity(entity_group, entity_text):
                            all_entities.append(entity_text)
                
                # # Remove duplicates while preserving order
                # unique_entities = []
                # seen = set()
                # for entity in all_entities:
                #     if entity not in seen:
                #         unique_entities.append(entity)
                #         seen.add(entity)
                
                output.append(all_entities)
                
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                output.append([])
        
        return output

    def _is_valid_entity(self, entity_group: str, entity_text: str) -> bool:
        """Check if entity is valid based on label type and requirements"""
        
        # Map label types to entity groups
        label_mapping = {
            "persons": ["PER", "B-PER", "I-PER"],
            "organizations": ["ORG", "B-ORG", "I-ORG"],
            "locations": ["LOC", "B-LOC", "I-LOC"],
            "misc": ["MISC", "B-MISC", "I-MISC"]
        }
        
        # Check if entity group matches the label type
        valid_groups = label_mapping.get(self.label, [])
        if entity_group not in valid_groups:
            return False
        
        # Check full name requirement
        if self.require_full_name:
            return len(entity_text.split()) > 1
        
        return True

    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"message": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__,
            "label": self.label,
            "require_full_name": self.require_full_name,
            "device": "GPU" if self.device == 0 else "CPU"
        }




if __name__ == "__main__":
    # Test the extractor
    
    
    # Test single entity type extractor
    test_extractor(
        extractor=PretrainedBERTEntityExtractor(label="persons"),
        extractor_multi=PretrainedBERTEntityExtractor(label="organizations"),
        extractor_multi_many=PretrainedBERTEntityExtractor(label="locations")
    )

