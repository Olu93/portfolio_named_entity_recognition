from port.entity_extractor import SingleEntityExtractor
from utils.develop import test_extractor
from utils.typings import TextInput
import spacy
import spacy.cli
import logging

logger = logging.getLogger(__name__)


def load_spacy_model(model_name: str = "en_core_web_sm"):
    logger.info(f"Loading spacy model '{model_name}'")
    try:
        nlp = spacy.load(model_name)
        logger.info(f"Loaded spacy model '{model_name}'")
        return nlp
    except OSError:
        logger.info(f"Downloading spacy model '{model_name}'")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
        logger.info(f"Loaded spacy model '{model_name}'")
        return nlp

class SpacyEntityExtractor(SingleEntityExtractor):
    def __init__(
        self,
        model: str = "en_core_web_sm",
        labels: list[str] = ["PERSON"],
        require_full_name: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nlp = load_spacy_model(model)

        self.labels = set(labels)
        self.require_full_name = require_full_name

    def _fit(self, X: TextInput, y: TextInput = None):
        return self  # pretrained, no fitting needed

    def _predict(self, X: TextInput):
        output: list[list[str]] = []
        for text in X:
            doc = self.nlp(text)
            ents = [
                ent.text
                for ent in doc.ents
                if ent.label_ in self.labels
                and (not self.require_full_name or len(ent.text.split()) > 1)
            ]
            output.append(ents)
        return output
    
class FastSpacyEntityExtractor(SpacyEntityExtractor):
    """
    Better version of sliding window extractor and spacy entity extractor.
    Combines all the entities into a single list and then uses the sliding window method to extract the entities.
    """

    def _fit(self, X:TextInput, y:TextInput=None):
        self.entities = {entity: entity.split() for entities in y for entity in entities}
        return self

    def _predict(self, X: TextInput):
        output: list[list[str]] = []
        final_result = []
        for text in X:
            result = []
            doc = self.nlp(text)
            ents = [
                ent.text
                for ent in doc.ents
                if ent.label_ 
            ]

            entity_words = " ".join(ents).split()
            for entity, entity_split in self.entities.items():
                partial_result = []
                for i in range(len(entity_words) - len(entity_split) + 1):
                    current_window = entity_words[i:i+len(entity_split)]
                    if tuple(current_window) == tuple(entity_split):    
                        partial_result.append(current_window)
                result.extend([" ".join(t) for t in partial_result])
            final_result.append(result)
        return final_result
    
class FastestSpacyEntityExtractor(FastSpacyEntityExtractor):
    """
    Better version of sliding window extractor and spacy entity extractor.
    Combines all the entities into a single list and then uses the sliding window method to extract the entities.
    """

    def _fit(self, X:TextInput, y:TextInput=None):
        self.entities = {entity: entity.split() for entities in y for entity in entities}
        return self

    def _predict(self, X: TextInput):
        output: list[list[str]] = []
        final_result = []
        for text in X:
            result = []
            doc = self.nlp(text)
            ents = {
                ent.text:ent.label_
                for ent in doc.ents
                if ent.label_ 
            }

            for entity, _ in self.entities.items():
                if entity in ents:
                    result.append(entity)
                    continue

            final_result.append(result)
        return final_result

if __name__ == "__main__":
    # extract persons by default
    test_extractor(
        extractor=SpacyEntityExtractor(labels=["PERSON"]),
        extractor_multi=SpacyEntityExtractor(labels=["PERSON"]),
        extractor_multi_many=SpacyEntityExtractor(labels=["PERSON"])
    )

    print("\n\n DIFFERENT EXTRACTOR\n\n")

    test_extractor(
        extractor=FastSpacyEntityExtractor(labels=["PERSON"]),
        extractor_multi=FastSpacyEntityExtractor(labels=["PERSON"]),
        extractor_multi_many=FastSpacyEntityExtractor(labels=["PERSON"])
    )

    print("\n\n DIFFERENT EXTRACTOR\n\n")

    test_extractor(
        extractor=FastestSpacyEntityExtractor(labels=["PERSON"]),
        extractor_multi=FastestSpacyEntityExtractor(labels=["PERSON"]),
        extractor_multi_many=FastestSpacyEntityExtractor(labels=["PERSON"])
    )
