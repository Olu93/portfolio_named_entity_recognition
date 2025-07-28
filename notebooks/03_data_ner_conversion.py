# %% 
from notebook_config import FILES_DIR
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal
from tqdm import tqdm

# %% Define Pydantic schema
class TokenNERPair(BaseModel):
    token: str
    tag: Literal["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "O"]

class Entity(BaseModel):
    entity_type: Literal["PER", "ORG", "LOC", "MISC"] = Field(description="The type of entity person=PER, organization=ORG, location=LOC, other=MISC")
    entity_value: str = Field(description="The value of the entity")

class NERAnnotation(BaseModel):
    original_text: str
    tokens: List[TokenNERPair]
    entities: list[Entity] = Field(description="List of entities found in the text")

# %% Load data
input_file = FILES_DIR / "semantic_split_complete_dataset.csv"
df = pd.read_csv(input_file).iloc[638:]
lines = df['text'].tolist()
print(f"Found {len(lines)} lines to process from CSV")

# %% LLM and parser
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0).with_structured_output(NERAnnotation)
# parser = PydanticOutputParser(pydantic_object=NERAnnotation)

# %% PromptTemplate version
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
You are an expert Named Entity Recognition (NER) annotator.
Annotate the given text in CoNLL-2003 format using BIO tagging.

Entity types to identify:
- PER: Names of people. ⚠️ ONLY include **full names** (e.g., "Barack Obama"). ❌ Do NOT include:
  - First names only (e.g., "Barack")
  - Last names only (e.g., "Obama")
  - Initials or abbreviations (e.g., "B. Obama")
  - Titles or roles (e.g., "President", "Dr.", "CEO")
  - Name fragments or mentions without a clear full name

- ORG: Companies, institutions, government bodies (e.g., Microsoft, United Nations, FBI)
- LOC: Countries, cities, states, physical locations (e.g., Germany, New York, Kremlin)
- MISC: Other named entities that don't fit the above categories

BIO tagging:
Use standard CoNLL-2003 tags:
B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O

Reference examples:
PERSONS: Joe Biden, Angela Merkel, Elon Musk
ORGANIZATIONS: Google, NATO, European Commission
LOCATIONS: Paris, Brazil, Mount Everest

Known entities in this text:
- PERSONS: {persons}
- ORGANIZATIONS: {organizations}
- LOCATIONS: {locations}

Now annotate the text accordingly.

Text to annotate: {text}
""")


# %% Combine into chain
chain = prompt_template | llm

# %% Process articles one by one
results = []
output_file = FILES_DIR / "ner_annotations_2.conll"
json_file = FILES_DIR / "ner_annotations_2.json"
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
            # "format_instructions": parser.get_format_instructions()
        }
        
        # Process single article
        parsed_result = chain.invoke(input_dict)
        
        if parsed_result:
            # Process the chunk directly
            tokens_and_tags = [(t.token, t.tag) for t in parsed_result.tokens]
            all_entities = [e.model_dump(mode="python") for e in parsed_result.entities]

            result = {
                "original_text": line,
                "conll_annotations": tokens_and_tags,
                "entities": all_entities
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
            "entities": [],
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