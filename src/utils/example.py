import json

from constants import EXAMPLES_PATH, MISC_DIR


class SingleExample:

    def __init__(self, text: str, person_entities: list[str], organization_entities: list[str], location_entities: list[str]):
        self.text = text
        self.person_entities = person_entities
        self.organization_entities = organization_entities
        self.location_entities = location_entities


class DefaultExample(SingleExample):
    def __init__(self):
        text: str = "A federal judge has ruled against House Republicans who tried to challenge security screening on Capitol Hill for members of Congress. Reps. Louie Gohmert of Texas, Andrew Clyde of Georgia and Lloyd Smucker of Pennsylvania were fined thousands of dollars each by the sergeant at arms after they skipped security screenings outside the House chamber that were put in place following the January 6, 2021, attack on the US Capitol. The trio then sued in DC District Court to challenge the House rules. But Judge Timothy Kelly, a Trump appointee, on Monday dismissed the case, saying he did not have jurisdiction. The House sergeant at arms and the House’s top administrator were protected from the court wading into the House rules because of the Constitution’s Speech or Debate Clause, the judge determined.  “Here, each challenged act of the House Officers qualifies as a legislative act,” Kelly wrote. “Thus, the Speech or Debate Clause bars the Members’ claims.”"
        person_entities: list[str] = ["Louie Gohmert", "Timothy Kelly", "Andrew Clyde", "Lloyd Smucker"]
        organization_entities: list[str] = ["DC District Court"]
        location_entities: list[str] = ["Georgia", "Pennsylvania", "Texas"]
        super().__init__(text, person_entities, organization_entities, location_entities)

class MultiExample:
    def __init__(self, examples_path: str = EXAMPLES_PATH):
        self.examples = json.load(open(examples_path))

    def __getitem__(self, index: int):
        text, person_entities, organization_entities, location_entities = self.examples[index]['text'], self.examples[index]['persons'], self.examples[index]['organizations'], self.examples[index]['locations']
        return SingleExample(text, person_entities, organization_entities, location_entities)


class MultiExampleMany(MultiExample):
    def __init__(self, examples_path: str = MISC_DIR / 'examples_many.json'):
        self.examples = json.load(open(examples_path))






