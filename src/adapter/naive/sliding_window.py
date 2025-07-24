
from port.entity_extractor import SingleEntityExtractor
from utils.develop import test_extractor
from utils.typings import TextInput
import re

class SlidingWindowExtractor(SingleEntityExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _fit(self, X:TextInput, y:TextInput=None):
        self.entities = {entity: entity.split() for entities in y for entity in entities}
        return self

    
    def _predict(self, X:TextInput):
        # Remove all non-alphanumeric characters
        X_ = [re.sub(r'[^a-zA-Z0-9\s]', '', text) for text in X]
        final_result = []
        # For every entity sliding window over the text and check if the entity is found
        for text in X_:
            result = []
            text_split = text.split()
            for entity, entity_split in self.entities.items():
                partial_result = []
                for i in range(len(text_split) - len(entity_split) + 1):
                    current_window = text_split[i:i+len(entity_split)]
                    if tuple(current_window) == tuple(entity_split):    
                        partial_result.append(current_window)
                result.extend([" ".join(t) for t in partial_result])
            final_result.append(result)
        return final_result


# Main function to test the SlidingWindowExtractor by using simple sentence and instantiating model, then fitting and then running a prediction
if __name__ == "__main__":
    test_extractor(extractor=SlidingWindowExtractor(), extractor_multi=SlidingWindowExtractor())
