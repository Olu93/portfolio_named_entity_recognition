# %% [markdown]
# # Named Entity Recognition (NER) Data Conversion
# 
# This notebook converts the chunked dataset into CoNLL-2003 format using LLM-based annotation. We'll use GPT-4o-mini to automatically annotate entities in the text chunks, creating a structured dataset suitable for training and evaluating NER models.
# 
# ## Setup: Import required libraries and dependencies

# %%
from notebook_config import DATASETS_DIR, FILES_DIR
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal
from tqdm import tqdm

# %% [markdown]
# ## Define Pydantic Schema for Structured Output
# 
# Create Pydantic models to ensure consistent and validated output from the LLM:
# 
# **TokenNERPair**: Represents individual token-tag pairs in BIO tagging format
# **Entity**: Represents extracted entities with their type and value
# **NERAnnotation**: Complete annotation structure containing original text, tokenized annotations, and entity list
# 
# This structured approach ensures data consistency and enables easy validation of LLM outputs.

# %%
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

# %% [markdown]
# ## Load Chunked Dataset
# 
# Load the semantically chunked dataset created in the previous notebook. This dataset contains:
# - **Text chunks**: Optimally-sized text segments for NER annotation
# - **Entity metadata**: Pre-existing entity annotations from the original dataset
# - **Token counts**: Pre-computed token lengths for each chunk
# 
# We'll use this as input for LLM-based NER annotation to create training data.

# %%
input_file = DATASETS_DIR / "semantic_split_complete_dataset.csv"
df = pd.read_csv(input_file).iloc[:5]
lines = df['text'].tolist()
print(f"Found {len(lines)} lines to process from CSV")

# %% [markdown]
# ## Initialize LLM and Structured Output Parser
# 
# Set up the LLM with structured output capabilities:
# - **Model**: GPT-4o-mini for cost-effective yet accurate annotation
# - **Temperature**: 0.0 for consistent, deterministic outputs
# - **Structured output**: Enforces Pydantic schema validation
# 
# This ensures reliable and consistent NER annotations across all text chunks.

# %%
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0).with_structured_output(NERAnnotation)
# parser = PydanticOutputParser(pydantic_object=NERAnnotation)

# %% [markdown]
# ## Create Specialized NER Prompt Template
# 
# Design a comprehensive prompt template that guides the LLM to produce high-quality NER annotations:
# 
# **Key Features:**
# - **Entity type definitions**: Clear specifications for PER, ORG, LOC, and MISC entities
# - **BIO tagging instructions**: Standard CoNLL-2003 format with proper begin/inside/outside tags
# - **Quality constraints**: Specific rules for person names (full names only, no titles)
# - **Reference examples**: Concrete examples to guide annotation decisions
# - **Known entities**: Incorporates pre-existing entity annotations as context
# 
# This prompt ensures consistent annotation quality and adherence to NER standards.

# %%
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


# %% [markdown]
# ## Create Processing Chain
# 
# Combine the prompt template with the LLM to create a processing chain that:
# - Takes text input and entity context
# - Applies the NER annotation prompt
# - Returns structured output conforming to our Pydantic schema
# 
# This chain ensures consistent processing across all text chunks.

# %%
chain = prompt_template | llm

# %% [markdown]
# ## Initialize Output Files and Processing Loop
# 
# Set up output files for both CoNLL format and JSON format:
# - **CoNLL file**: Standard format for NER model training and evaluation
# - **JSON file**: Rich format preserving full annotation details and metadata
# 
# Initialize empty files to prepare for incremental writing during processing.

# %%
results = []
output_file = FILES_DIR / "ner_annotations.conll"
json_file = FILES_DIR / "ner_annotations.json"
with open(output_file, 'w', encoding='utf-8'): pass  # Empty file

# %% [markdown]
# ## Process Text Chunks with LLM Annotation
# 
# Iterate through each text chunk and apply LLM-based NER annotation:
# 
# **Processing Steps:**
# 1. **Input preparation**: Extract text and entity context from each row
# 2. **LLM annotation**: Apply the NER chain to generate structured annotations
# 3. **Output formatting**: Convert to both CoNLL and JSON formats
# 4. **Incremental saving**: Write results after each chunk to prevent data loss
# 5. **Error handling**: Capture and log any processing errors
# 
# This approach ensures robust processing of large datasets with progress tracking and error recovery.

# %%
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

# %% [markdown]
# ## Processing Summary and Statistics
# 
# Generate comprehensive statistics about the annotation process:
# 
# **Key Metrics:**
# - **Total processed**: Number of chunks successfully annotated
# - **Total tokens**: Overall token count across all annotations
# - **Tag distribution**: Frequency of each entity type and BIO tag
# - **Error analysis**: Any processing failures and their causes
# 
# These statistics help validate annotation quality and identify potential issues in the dataset.

# %%
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

# %% [markdown]
# ## Sample Annotation Review
# 
# Display sample annotations to validate the quality and format of the generated NER data:
# 
# **Review Elements:**
# - **Original text**: Source content being annotated
# - **CoNLL format**: Token-tag pairs in standard format
# - **Entity extraction**: Structured entity information
# 
# This review helps ensure the annotation quality meets expectations for model training.

# %%
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