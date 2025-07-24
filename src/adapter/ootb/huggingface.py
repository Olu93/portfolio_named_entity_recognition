from port.entity_extractor import SingleEntityExtractor
from utils.develop import test_extractor
from utils.typings import TextInput
from transformers import pipeline


class HuggingFaceEntityExtractor(SingleEntityExtractor):
    def __init__(
        self,
        model: str = "dslim/bert-base-NER",
        labels: list[str] = ["PER"],
        require_full_name: bool = True,
        aggregation_strategy: str = "simple",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # HuggingFace NER pipeline groups subword tags into spans
        self.ner = pipeline(
            "ner",
            model=model,
            aggregation_strategy=aggregation_strategy
        )
        self.labels = set(labels)
        self.require_full_name = require_full_name

    def _fit(self, X: TextInput, y: TextInput = None):
        return self  # no training

    def _predict(self, X: TextInput):
        output: list[list[str]] = []
        for text in X:
            spans = self.ner(text)
            ents = [
                span["word"]
                for span in spans
                if span["entity_group"] in self.labels
                   and (not self.require_full_name or len(span["word"].split()) > 1)
            ]
            output.append(ents)
        return output


class FastHuggingFaceEntityExtractor(HuggingFaceEntityExtractor):
    """
    Similar to FastSpacyEntityExtractor: filter only those ground-truth entities
    found in the HF spans, rather than repeating sliding-window on full text.
    """
    def _fit(self, X: TextInput, y: TextInput = None):
        self.entities = {e: e.split() for docs in y for e in docs}
        return self

    def _predict(self, X: TextInput):
        output: list[list[str]] = []
        for text in X:
            spans = self.ner(text)
            found = {span["word"] for span in spans if span["entity_group"] in self.labels}
            result = [e for e in self.entities if e in found]
            output.append(result)
        return output


if __name__ == "__main__":
    # single‐type PERSON
    test_extractor(
        extractor=HuggingFaceEntityExtractor(labels=["PER"]),
        extractor_multi=HuggingFaceEntityExtractor(labels=["PER"]),
        extractor_multi_many=HuggingFaceEntityExtractor(labels=["PER"])
    )

    print("\n\nFAST VARIANT\n\n")

    test_extractor(
        extractor=FastHuggingFaceEntityExtractor(labels=["PER"]),
        extractor_multi=FastHuggingFaceEntityExtractor(labels=["PER"]),
        extractor_multi_many=FastHuggingFaceEntityExtractor(labels=["PER"])
    )
