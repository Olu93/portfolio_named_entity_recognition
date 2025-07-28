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
    MAP = {
        "persons": ["PERSON"],
        "organizations": ["ORG"],
        "locations": ["LOC", "GPE"]
    }

    def __init__(
        self,
        model: str = "en_core_web_sm",
        label: str = "persons",
        require_full_name: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nlp = load_spacy_model(model)

        self.labels = self.MAP[label]
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
    


if __name__ == "__main__":
    # extract persons by default
    test_extractor(
        extractor=SpacyEntityExtractor(label="persons"),
        extractor_multi=SpacyEntityExtractor(label="persons"),
        extractor_multi_many=SpacyEntityExtractor(label="persons")
    )

    print("\n\n DIFFERENT EXTRACTOR\n\n")

    test_extractor(
        extractor=FastSpacyEntityExtractor(label="persons"),
        extractor_multi=FastSpacyEntityExtractor(label="persons"),
        extractor_multi_many=FastSpacyEntityExtractor(label="persons")
    )

    print("\n\n DIFFERENT EXTRACTOR\n\n")

    test_extractor(
        extractor=FastestSpacyEntityExtractor(label="persons"),
        extractor_multi=FastestSpacyEntityExtractor(label="persons"),
        extractor_multi_many=FastestSpacyEntityExtractor(label="persons")
    )
