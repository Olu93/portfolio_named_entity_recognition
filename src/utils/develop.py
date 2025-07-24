from port.entity_extractor import SingleEntityExtractor
from utils.example import DefaultExample, MultiExample, MultiExampleMany
import logging


def test_extractor(extractor: SingleEntityExtractor = None, extractor_multi: SingleEntityExtractor = None, extractor_multi_many: SingleEntityExtractor = None):
    # Simple sentence to test the SlidingWindowExtractor
    logger = logging.getLogger(__name__)
    
    if extractor is not None:
        logger.info(f" =============================== Starting the {extractor.__class__.__name__} - Single Example =============================== ")
        sentences = DefaultExample().text
        person_entities = DefaultExample().person_entities
        # Instantiate the SlidingWindowExtractor
        extractor = extractor
        # Fit the extractor
        extractor.fit(sentences, person_entities)
        # Run a prediction
        logger.info(f"PREDICTION: {extractor.predict(sentences)}")
        logger.info(f"TRAINING TIME: {extractor.stats['training']['time']}")
        logger.info(f"INFERENCE TIME: {extractor.stats['inference']['time']}")
        logger.info(f"METRICS: {extractor.stats['evaluation']['metrics']}")
        logger.info(f" =============================== Ending the {extractor.__class__.__name__} - Single Example =============================== \n\n")

    if extractor_multi is not None:
        logger.info(f" =============================== Starting the {extractor_multi.__class__.__name__} - Multi Example =============================== ")
        examples = MultiExample()

        sentences = [example.text for example in examples]
        person_entities = [example.person_entities for example in examples]
        
        extractor = extractor_multi
        extractor.fit(sentences, person_entities)
        logger.info(f"PREDICTION: {extractor.predict(sentences)}")
        logger.info(f"TRAINING TIME: {extractor.stats['training']['time']}")
        logger.info(f"INFERENCE TIME: {extractor.stats['inference']['time']}")
        logger.info(f"METRICS: {extractor.stats['evaluation']['metrics']}")
        logger.info(f" =============================== Ending the {extractor_multi.__class__.__name__} - Multi Example =============================== ")

    if extractor_multi_many is not None:
        logger.info(f" =============================== Starting the {extractor_multi.__class__.__name__} - Multi-Many Example =============================== ")
        examples = MultiExampleMany()

        sentences = [example.text for example in examples]
        person_entities = [example.person_entities for example in examples]
        
        extractor = extractor_multi
        extractor.fit(sentences, person_entities)
        logger.info(f"PREDICTION: {extractor.predict(sentences)}")
        logger.info(f"TRAINING TIME: {extractor.stats['training']['time']}")
        logger.info(f"INFERENCE TIME: {extractor.stats['inference']['time']}")
        logger.info(f"METRICS: {extractor.stats['evaluation']['metrics']}")
        logger.info(f" =============================== Ending the {extractor_multi.__class__.__name__} - Multi-Many Example =============================== ")
