# %% 
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from tqdm import tqdm
import time
from notebooks import FILES_DIR

# %%
# Read the input CSV file
input_file = FILES_DIR / "full_data_clean_finetune_2.csv"
df = pd.read_csv(input_file)
lines = df['text'].tolist()

print(f"Found {len(lines)} lines to process from CSV")

# %%
# Initialize the LLM
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.0
)

# %%
# Create the prompt for CoNLL-2003 format with examples
system_prompt = """You are an expert Named Entity Recognition (NER) annotator. 
Annotate the given text in CoNLL-2003 format with BIO tagging.

Entity types to identify:
- PER: Names of people (full names ONLY. Not just first names or last names)
- ORG: Companies, institutions, government bodies, etc.
- LOC: Countries, cities, states, addresses, etc.
- MISC: Other named entities that don't fit the above categories

BIO tagging:
- B-PER: Beginning of a person name
- I-PER: Inside of a person name
- B-ORG: Beginning of an organization name
- I-ORG: Inside of an organization name
- B-LOC: Beginning of a location name
- I-LOC: Inside of a location name
- B-MISC: Beginning of a miscellaneous entity
- I-MISC: Inside of a miscellaneous entity
- O: Outside of any entity

Examples of entities to recognize:

PERSONS (PER):
- Joe Biden, Donald Trump, Barack Obama, Vladimir Putin, Xi Jinping
- John Smith, Mary Johnson, Dr. Anthony Fauci, President Biden

ORGANIZATIONS (ORG):
- White House, United Nations, NATO, European Union, World Health Organization
- Microsoft, Apple, Google, Amazon, Tesla, Facebook
- Department of Defense, Central Intelligence Agency, Federal Bureau of Investigation
- Universities: Harvard University, Stanford University, MIT
- News organizations: CNN, BBC, Reuters, Associated Press

LOCATIONS (LOC):
- Countries: United States, Russia, China, Germany, France, Japan
- Cities: New York, London, Paris, Tokyo, Moscow, Beijing
- States: California, Texas, Florida, New York, Texas
- Landmarks: White House, Eiffel Tower, Statue of Liberty, Kremlin

Return format: Each token on a new line with its BIO tag, separated by space.

Example:
Input: "Joe Biden visited the White House in Washington DC."
Output:
Joe B-PER
Biden I-PER
visited O
the O
White B-ORG
House I-ORG
in O
Washington B-LOC
DC I-LOC
. O

If no entities found, tag all tokens as O."""

# %%
# Process the lines in CoNLL-2003 format with on-the-fly saving
results = []
output_file = FILES_DIR / "ner_annotations_2.conll"
json_file = FILES_DIR / "ner_annotations_2.json"

# Create/clear the files
with open(output_file, 'w', encoding='utf-8') as f:
    pass  # Create empty file

for i, line in enumerate(tqdm(lines, desc="Processing lines")):
    try:
        # Get the actual entities for this line from the CSV
        row = df.iloc[i]
        actual_persons = row.get('persons', '')
        actual_organizations = row.get('organizations', '')
        actual_locations = row.get('locations', '')
        
        # Create entity guidance message
        entity_guidance = f"""For this specific text, the expected entities are:
- PERSONS: {actual_persons}
- ORGANIZATIONS: {actual_organizations} 
- LOCATIONS: {actual_locations}

Please annotate the text accordingly, focusing on these specific entities."""
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=entity_guidance),
            HumanMessage(content=line)
        ]
        
        # Get response
        response = llm.invoke(messages)
        
        # Parse CoNLL format from response
        conll_lines = response.content.strip().split('\n')
        tokens_and_tags = []
        
        for conll_line in conll_lines:
            if conll_line.strip() and ' ' in conll_line:
                parts = conll_line.strip().split()
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[1]
                    tokens_and_tags.append((token, tag))
        
        # Store result
        result = {
            "original_text": line,
            "conll_annotations": tokens_and_tags
        }
        results.append(result)
        
        # Save CoNLL format on the fly
        with open(output_file, 'a', encoding='utf-8') as f:
            for token, tag in tokens_and_tags:
                f.write(f"{token} {tag}\n")
            f.write("\n")  # Empty line between sentences
        
        # Save JSON on the fly (append to list)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
        
    except Exception as e:
        print(f"Error processing line {i}: {str(e)}")
        result = {
            "original_text": line,
            "conll_annotations": [],
            "error": str(e)
        }
        results.append(result)
        
        # Save error result to JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# %%
# Results are already saved on the fly
print(f"CoNLL-2003 results saved to {output_file}")
print(f"JSON results saved to {json_file}")
print(f"Processing completed! Processed {len(results)} lines.")

# %%
# Print summary
total_tokens = sum(len(result["conll_annotations"]) for result in results if "error" not in result)
print(f"Total tokens processed: {total_tokens}")

# Count by tag type
tag_counts = {}
for result in results:
    if "error" not in result:
        for token, tag in result["conll_annotations"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

print("Tags by type:")
for tag, count in sorted(tag_counts.items()):
    print(f"  {tag}: {count}")

# %%
# Show some examples
print("\nExample CoNLL-2003 annotations:")
for i, result in enumerate(results[:2]):
    if "error" not in result:
        print(f"\n{i+1}. Text: {result['original_text'][:100]}...")
        print("   CoNLL format:")
        for token, tag in result["conll_annotations"][:10]:  # Show first 10 tokens
            print(f"     {token} {tag}")
        if len(result["conll_annotations"]) > 10:
            print("     ...")
# %%
