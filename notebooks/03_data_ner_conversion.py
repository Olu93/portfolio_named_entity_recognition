# %% 
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal
from tqdm import tqdm
import time
from notebooks import FILES_DIR

# %% Define Pydantic schema
class TokenNERPair(BaseModel):
    token: str
    tag: Literal["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "O"]

class Entity(BaseModel):
    entity_type: Literal["PER", "ORG", "LOC", "MISC"] = Field(description="The type of entity person=PER, organization=ORG, location=LOC, other=MISC")
    entity_value: str = Field(description="The value of the entity")

class NERAnnotatedSentence(BaseModel):
    original_text: str
    tokens: List[TokenNERPair]
    entities: list[Entity] = Field(description="List of entities found in the text")

class NERAnnotation(BaseModel):
    sentences: List[NERAnnotatedSentence]

# %% Load data
input_file = FILES_DIR / "semantic_split_complete_dataset.csv"
df = pd.read_csv(input_file)
lines = df['text'].tolist()
print(f"Found {len(lines)} lines to process from CSV")

# %% LLM and parser
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
parser = PydanticOutputParser(pydantic_object=NERAnnotation)

# %% PromptTemplate version
prompt_template = PromptTemplate.from_template("""
You are an expert Named Entity Recognition (NER) annotator. 
Annotate the given text in CoNLL-2003 format with BIO tagging.

Entity types to identify:
- PER: Names of people (full names ONLY)
- ORG: Companies, institutions, government bodies, etc.
- LOC: Countries, cities, states, addresses, etc.
- MISC: Other named entities that don't fit the above categories

BIO tagging:
- B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O

Examples of entities:
PERSONS: Joe Biden, Anthony Fauci, Vladimir Putin
ORGANIZATIONS: United Nations, Microsoft, FBI
LOCATIONS: United States, New York, Kremlin

{format_instructions}

For this text, these are example entities:
- PERSONS: {persons}
- ORGANIZATIONS: {organizations}
- LOCATIONS: {locations}

Please annotate the text accordingly.

Text to annotate: {text}
""")

# %% Combine into chain
chain = prompt_template | llm | parser

# %% Process articles one by one
results = []
output_file = FILES_DIR / "ner_annotations.conll"
json_file = FILES_DIR / "ner_annotations.json"
with open(output_file, 'w', encoding='utf-8'): pass  # Empty file

# %% Process one by one
for i, (line, row) in enumerate(tqdm(zip(lines, df.itertuples()), desc="Processing articles", total=len(lines))):
    try:
        # Prepare input for single article
        input_dict = {
            "text": line,
            "persons": getattr(row, 'persons', ''),
            "organizations": getattr(row, 'organizations', ''),
            "locations": getattr(row, 'locations', ''),
            "format_instructions": parser.get_format_instructions()
        }
        
        # Process single article
        parsed_result = chain.invoke(input_dict)
        
        if parsed_result and parsed_result.sentences:
            # Process all sentences in the chunk
            tokens_and_tags = []
            all_entities = []
            annotated_sentences = []
            
            for sentence in parsed_result.sentences:
                sentence_tokens = [(t.token, t.tag) for t in sentence.tokens]
                tokens_and_tags.extend(sentence_tokens)
                all_entities.extend(sentence.entities)
                annotated_sentences.append(sentence.model_dump(mode="python"))

            result = {
                "original_text": line,
                "conll_annotations": tokens_and_tags,
                "annotated_sentences": annotated_sentences,
                "all_entities": [e.model_dump(mode="python") for e in all_entities]
            }
        else:
            raise ValueError("Empty or malformed result")

        results.append(result)

        # Write to CoNLL file
        with open(output_file, 'a', encoding='utf-8') as f:
            for token, tag in tokens_and_tags:
                f.write(f"{token} {tag}\n")
            f.write("\n")
            
        # Save JSON after each article (incremental saving)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error processing article {i}: {str(e)}")
        results.append({
            "original_text": line,
            "conll_annotations": [],
            "annotated_sentences": [],
            "all_entities": [],
            "error": str(e)
        })
        
        # Save error results to JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# %% Summary
print(f"\n✅ CoNLL saved to {output_file}")
print(f"✅ JSON saved to {json_file}")
print(f"Total processed: {len(results)}")

total_tokens = sum(len(r["conll_annotations"]) for r in results if "error" not in r)
print(f"Total tokens processed: {total_tokens}")

tag_counts = {}
for result in results:
    if "error" not in result:
        for _, tag in result["conll_annotations"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

print("Tags by type:")
for tag, count in sorted(tag_counts.items()):
    print(f"  {tag}: {count}")

# %% Show sample
print("\nExample annotations:")
for i, r in enumerate(results[:2]):
    if "error" not in r:
        print(f"\n{i+1}. Text: {r['original_text'][:100]}...")
        print("   CoNLL:")
        for token, tag in r["conll_annotations"][:10]:
            print(f"     {token} {tag}")
        if len(r["conll_annotations"]) > 10:
            print("     ...")
# %%