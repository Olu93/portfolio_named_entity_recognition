from adapter.finetuning.pretrained import PretrainedBERTEntityExtractor
from adapter.ootb.huggingface import HuggingFaceEntityExtractor
from adapter.ootb.spacy import SpacyEntityExtractor
from adapter.ootb.llm import LangChainEntityExtractor
from adapter.naive.sliding_window import SlidingWindowExtractor
from constants import FILES_DIR

# Configuration for all models
MODEL_CONFIGS = [
    {
        "name": "PretrainedBERTEntityExtractor",
        "extra_info": {
            "description": "Pretrained BERT model for entity extraction",
            "model": "bert-base-cased-finetuned",
            "type": "transformer",
        },
        "persons": {
            "extractor": PretrainedBERTEntityExtractor,
            "params": {
                "model_path": FILES_DIR / "pretrained" / "bert_ner_finetuned",
                "label": "persons",
            },
        },
        "organizations": {
            "extractor": PretrainedBERTEntityExtractor,
            "params": {
                "model_path": FILES_DIR / "pretrained" / "bert_ner_finetuned",
                "label": "organizations",
            },
        },
        "locations": {
            "extractor": PretrainedBERTEntityExtractor,
            "params": {
                "model_path": FILES_DIR / "pretrained" / "bert_ner_finetuned",
                "label": "locations",
            },
        },
    },
    {
        "name": "PretrainedBERTEntityExtractor",
        "extra_info": {
            "description": "Pretrained BERT model for entity extraction",
            "model": "dslim-bert-base-cased-finetuned",
            "type": "transformer",
        },
        "persons": {
            "extractor": PretrainedBERTEntityExtractor,
            "params": {
                "model_path": FILES_DIR / "pretrained" / "dslim_bert_ner_finetuned",
                "label": "persons",
            },
        },
        "organizations": {
            "extractor": PretrainedBERTEntityExtractor,
            "params": {
                "model_path": FILES_DIR / "pretrained" / "dslim_bert_ner_finetuned",
                "label": "organizations",
            },
        },
        "locations": {
            "extractor": PretrainedBERTEntityExtractor,
            "params": {
                "model_path": FILES_DIR / "pretrained" / "dslim_bert_ner_finetuned",
                "label": "locations",
            },
        },
    },
    {
        "name": "HuggingFaceEntityExtractor",
        "extra_info": {
            "description": "BERT-based Named Entity Recognition model",
            "model": "dslim/bert-base-NER",
            "type": "transformer",
            "paper": "https://arxiv.org/abs/1810.04805",
        },
        "persons": {
            "extractor": HuggingFaceEntityExtractor,
            "params": {"model": "dslim/bert-base-NER", "label": "persons"},
        },
        "organizations": {
            "extractor": HuggingFaceEntityExtractor,
            "params": {"model": "dslim/bert-base-NER", "label": "organizations"},
        },
        "locations": {
            "extractor": HuggingFaceEntityExtractor,
            "params": {"model": "dslim/bert-base-NER", "label": "locations"},
        },
    },
    {
        "name": "HuggingFaceEntityExtractor",
        "extra_info": {
            "description": "DistilBERT model for Named Entity Recognition (faster and smaller than BERT)",
            "model": "dslim/distilbert-NER",
            "type": "transformer",
            "paper": "https://arxiv.org/abs/1910.01108",
        },
        "persons": {
            "extractor": HuggingFaceEntityExtractor,
            "params": {"model": "dslim/distilbert-NER", "label": "persons"},
        },
        "organizations": {
            "extractor": HuggingFaceEntityExtractor,
            "params": {"model": "dslim/distilbert-NER", "label": "organizations"},
        },
        "locations": {
            "extractor": HuggingFaceEntityExtractor,
            "params": {"model": "dslim/distilbert-NER", "label": "locations"},
        },
    },
    {
        "name": "HuggingFaceEntityExtractor",
        "extra_info": {
            "description": "BERT-large model fine-tuned on CoNLL-2003 English dataset",
            "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "type": "transformer",
            "paper": "https://arxiv.org/abs/1810.04805",
        },
        "persons": {
            "extractor": HuggingFaceEntityExtractor,
            "params": {
                "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "label": "persons",
            },
        },
        "organizations": {
            "extractor": HuggingFaceEntityExtractor,
            "params": {
                "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "label": "organizations",
            },
        },
        "locations": {
            "extractor": HuggingFaceEntityExtractor,
            "params": {
                "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "label": "locations",
            },
        },
    },
    # {
    #     "name": "HuggingFaceEntityExtractor",
    #     "extra_info": {
    #         "description": "RoBERTa-large model fine-tuned for NER on English text",
    #         "model": "Jean-Baptiste/roberta-large-ner-english",
    #         "type": "transformer",
    #         "paper": "https://arxiv.org/abs/1907.11692"
    #     },
    #     "persons": {
    #         "extractor": HuggingFaceEntityExtractor,
    #         "params": {
    #             "model": "Jean-Baptiste/roberta-large-ner-english",
    #             "label": "persons"
    #         }
    #     },
    #     "organizations": {
    #         "extractor": HuggingFaceEntityExtractor,
    #         "params": {
    #             "model": "Jean-Baptiste/roberta-large-ner-english",
    #             "label": "organizations"
    #         }
    #     },
    #     "locations": {
    #         "extractor": HuggingFaceEntityExtractor,
    #         "params": {
    #             "model": "Jean-Baptiste/roberta-large-ner-english",
    #             "label": "locations"
    #         }
    #     }
    # },
    {
        "name": "SlidingWindowExtractor",
        "extra_info": {
            "description": "Naive sliding window approach for entity extraction",
            "type": "rule-based",
        },
        "persons": {"extractor": SlidingWindowExtractor, "params": {}},
        "organizations": {"extractor": SlidingWindowExtractor, "params": {}},
        "locations": {"extractor": SlidingWindowExtractor, "params": {}},
    },
    {
        "name": "SpacyEntityExtractor",
        "extra_info": {
            "description": "SpaCy small English model for NER",
            "model": "en_core_web_sm",
            "type": "rule-based",
            "paper": "https://spacy.io/models/en",
        },
        "persons": {
            "extractor": SpacyEntityExtractor,
            "params": {"model": "en_core_web_sm", "label": "persons"},
        },
        "organizations": {
            "extractor": SpacyEntityExtractor,
            "params": {"model": "en_core_web_sm", "label": "organizations"},
        },
        "locations": {
            "extractor": SpacyEntityExtractor,
            "params": {"model": "en_core_web_sm", "label": "locations"},
        },
    },
    {
        "name": "SpacyEntityExtractor",
        "extra_info": {
            "description": "SpaCy medium English model for NER",
            "model": "en_core_web_md",
            "type": "rule-based",
            "paper": "https://spacy.io/models/en",
        },
        "persons": {
            "extractor": SpacyEntityExtractor,
            "params": {"model": "en_core_web_md", "label": "persons"},
        },
        "organizations": {
            "extractor": SpacyEntityExtractor,
            "params": {"model": "en_core_web_md", "label": "organizations"},
        },
        "locations": {
            "extractor": SpacyEntityExtractor,
            "params": {"model": "en_core_web_md", "label": "locations"},
        },
    },
    {
        "name": "SpacyEntityExtractor",
        "extra_info": {
            "description": "SpaCy large English model for NER",
            "model": "en_core_web_lg",
            "type": "rule-based",
            "paper": "https://spacy.io/models/en",
        },
        "persons": {
            "extractor": SpacyEntityExtractor,
            "params": {"model": "en_core_web_lg", "label": "persons"},
        },
        "organizations": {
            "extractor": SpacyEntityExtractor,
            "params": {"model": "en_core_web_lg", "label": "organizations"},
        },
        "locations": {
            "extractor": SpacyEntityExtractor,
            "params": {"model": "en_core_web_lg", "label": "locations"},
        },
    },
    {
        "name": "LangChainEntityExtractor",
        "extra_info": {
            "description": "OpenAI GPT-4o-nano model for entity extraction using LangChain",
            "model": "gpt-4o-nano",
            "type": "llm",
            "paper": "https://openai.com/research/gpt-4o",
        },
        "persons": {
            "extractor": LangChainEntityExtractor,
            "params": {"model_name": "gpt-4o-nano", "label": "PERSON"},
        },
        "organizations": {
            "extractor": LangChainEntityExtractor,
            "params": {"model_name": "gpt-4o-nano", "label": "ORG"},
        },
        "locations": {
            "extractor": LangChainEntityExtractor,
            "params": {"model_name": "gpt-4o-nano", "label": "LOC"},
        },
    },
] 