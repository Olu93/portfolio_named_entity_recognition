# Data Exploration
- coliumns are ['urls', 'text', 'persons', 'organizations', 'themes',
       'locations']
- Text contains broken dict with articlebody & html
- persons, organisations and locations need to be extracted
- persons and organisations are semicolon seperated and contains the named entity and an id.
- locations are comma seperated
- Data clean up is necessary. As well as dictionary of named entities
- Not clear what the number beside person and organisation represents


# Useful Regex

## Find inconsistent tags

### Simple
- (for\s)O(\n.+\s)(I-.{3})
- (and\s)O(\n.+\s)(I-.{3})
- (of\s)O(\n.+\s)(I-.{3})

#### Replacements
$1$3$2$3

### More specific
- ((&|to)\s)O(\n.+\s)(I-.{3})
#### Replacements
$2 $4$3$4


### RvW

Roe B-.{3}
v. O
Wade I-.{3}

#### Replacements
Roe B-PER
v. I-PER
Wade I-PER


### Short forms
\( O
(.+\s)I-(.{3})
\) O

( O
$1B-$2
) O


### Locations in front of organisations
(.+ B-)(.+)(\n.+ I-)(?!\2$)(.+) 
#### Replacement
$1ORG$3$4

### Locations in front of organisations 2
(.+ B-)(LOC)(\n.+ )(B-)(ORG)
#### Replacement
$1$5$3I-$5