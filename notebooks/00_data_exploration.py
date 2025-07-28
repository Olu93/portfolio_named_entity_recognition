# %% 
# Import notebooks package to set up the Python path

from notebook_config import FILES_DIR
import pandas as pd
import json
# %%
df = pd.read_csv(FILES_DIR / 'full_data.csv')
# %%
df.head()
# %%
df.tail()
# %%
df.info()
# %%
df.describe()
# %%
df.columns
# %%
# Columns are ['Unnamed: 0', 'urls', 'text', 'persons', 'organizations', 'themes', 'locations']
df.iloc[0]
# %%
df.iloc[0]['text']
# %%
# Looks likke 'Karen Jan Pierre,1533;Muhammad Ben Salman,706;Mohammed Ben Salman,55;Joe Biden,135'
df.iloc[0]['persons']

# %%
# Looks like 'White House,305;White House,1501;United Nations,174;United Nations,225;United Nations,1089;United Nations,1877;United States,320;United States,1292;United States,1418;United States,1516'
df.iloc[0]['organizations']
# %%
df.iloc[0]['themes']
# %%
# Looks like 'White House,Yemeni,United States,Yemen,Saudi Arabia'
df.iloc[0]['locations']
# %%
df.iloc[10]
# %%
df.iloc[10]['text']
# %%
# Count number of rows starting with \"articlebody\"
df[df['text'].str.startswith('\"articleBody\"')].shape[0]
# %%
# Count number of rows starting with <html
df[df['text'].str.startswith('<html')].shape[0]


# %%
# Take all people, organisations, split  by semilcolon and put id in a dict with list of elements which are the names
entities_dict = {
    'persons': {},
    'organizations': {},
    'locations': {}
}
for i in range(len(df)):
    persons = df.iloc[i]['persons']
    organizations = df.iloc[i]['organizations']
    locations = df.iloc[i]['locations']

    if not pd.isna(persons):
        for person in persons.split(';'):
            p, _id = person.split(',')
            if _id not in entities_dict['persons']:
                entities_dict['persons'][_id] = []
            entities_dict['persons'][_id].append(p)
    if not pd.isna(organizations):
        for organization in organizations.split(';'):
            o, _id = organization.split(',')
            if _id not in entities_dict['organizations']:
                entities_dict['organizations'][_id] = []
            entities_dict['organizations'][_id].append(o)
    if not pd.isna(locations):
        for location in locations.split(','):
            l, _id = location, str(hash(location))
            if _id not in entities_dict['locations']:
                entities_dict['locations'][_id] = []
            entities_dict['locations'][_id].append(l)
print(json.dumps(entities_dict, indent=4))
json.dump(entities_dict, open(FILES_DIR / 'misc' / 'entities_dict.json', 'w'), indent=4)
# %%
# Same as above but this time the name is the key and the value is the id we append to the  list
entities_reversed_dict = {
    'persons': {},
    'organizations': {},
    'locations': {}
}
for _type, elements in entities_dict.items():
    for k, list_of_entities in elements.items():
        for i in list_of_entities:
            if i not in entities_reversed_dict[_type]:
                entities_reversed_dict[_type][i] = []
            entities_reversed_dict[_type][i].append(k)
print(json.dumps(entities_reversed_dict, indent=4))
json.dump(entities_reversed_dict, open(FILES_DIR / 'misc' / 'entities_reversed_dict.json', 'w'), indent=4)
# %%
# %%
 