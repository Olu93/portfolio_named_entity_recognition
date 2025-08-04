from langchain_text_splitters import TextSplitter, TokenTextSplitter
import pandas as pd
import spacy
from utils.typings import TextInput
import logging
import time
import numpy as np
from pprint import pprint

logger = logging.getLogger(__name__)
def extract_objects(y:list[str]):
    final_result = []
    for obj in y:
        obj_result = []
        # Check if the object contains semicolons (case 1: elem,id;elem,id;elem,id)
        if ';' in obj:
            split_obj = obj.split(';')
            for e in split_obj:
                if e.strip():  # Skip empty strings
                    obj_result.append(e.split(',')[0])
        else:
            # Case 2: elem,elem,elem (just comma-separated)
            split_obj = obj.split(',')
            for e in split_obj:
                if e.strip():  # Skip empty strings
                    obj_result.append(e.strip())
        final_result.append(obj_result)
    return final_result
    

def convert_X_to_list(X: TextInput):
    start_time = time.time()
    final_result = []
    if isinstance(X, str):
        final_result.append(X)
    elif isinstance(X, pd.Series):
        final_result.extend(X.tolist())
    elif isinstance(X, pd.DataFrame):
        final_result.extend(X.values.tolist())
    elif isinstance(X, list) and all(isinstance(x, str) for x in X):
        final_result.extend(X)
    else:
        raise ValueError(f"Unsupported type: {type(X)}")
    logger.info(f"Converted to list of size {len(final_result)} in {time.time() - start_time} seconds")
    return final_result

def _replace_empty(e: list[str]):
    if isinstance(e, list) and len(e) == 0:
        return []
    if isinstance(e, list) and len(e) == 1:
        if e[0] == '':
            return []
    return e

def convert_y_to_list(y: TextInput):
    start_time = time.time()
    final_result = []
    if isinstance(y, str):
        final_result.extend([y])
    elif isinstance(y, pd.Series):
        # TODO: Cover cases in which the series elements are strings and not list
        final_result.extend([_replace_empty(e) for e in y.apply(lambda x: x.split(';')).tolist()])
    elif isinstance(y, pd.DataFrame):
        # TODO: Cover cases in which the dataframe elements are strings and not list
        final_result.extend([_replace_empty(e) for e in y.apply(lambda x: x.split(';')).values.tolist()])
    elif isinstance(y, list):
        all_data = []

        # In the case of list if the first element in y is a list then all the elements in y should be a list
        if isinstance(y[0], list):
            # If all elements in y are list then check if all elements within every list are strings
            
            for row_idx, row in enumerate(y):
                if isinstance(row, list):
                    all_data.extend([(row_idx, eidx, e, isinstance(e, str)) for eidx, e in enumerate(row)])
                else:
                    # If the element is not a list then raise a value error with the index of the element that is not a list
                    raise ValueError(f"Unsupported type: {type(y)} in row {row_idx}")
            all_data_array = np.array(all_data)
            if not np.all(all_data_array[:, 3:]):
                raise ValueError(f"Unsupported type: {type(y)} in {np.where(~np.array(all_data_array[:, 3:]))[0]}")
            final_result.extend(y)
        else:
            # If first element of y are not list then check if all elements are strings
            all_data = [(idx, e, isinstance(e, str)) for idx, e in enumerate(y)]
            all_data_array = np.array(all_data)
            if np.all(all_data_array[:, 2]):
                final_result.append(y)
            else:
                raise ValueError(f"Unsupported type: {type(y)} in {np.where(~np.array(all_data_array[:, 2]))[0]}")

    else:
        raise ValueError(f"Unsupported type: {type(y)}")
    logger.info(f"Converted to list of size {len(final_result)} in {time.time() - start_time} seconds")
    return final_result

def take_person_or_org(x: str):
    return x.split(',')[0]


class SpacySentenceSplitter(TextSplitter):
    def __init__(
        self,
        model: str = "en_core_web_sm",
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        length_function=None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                         length_function=length_function or len)
        self.nlp = spacy.load(model, disable=["ner", "tagger", "parser"])
        self.nlp.add_pipe('sentencizer')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    




# Test cases if run as main
if __name__ == "__main__":
    # TODO: Convert to pytest tests
    # Single example case
    splits = SpacySentenceSplitter().split_text("Americans are piling up credit card debt as they struggle to keep up with the high cost of living. US household debt surpassed $16 trillion for the first time ever during the second quarter, the New York Federal Reserve said Tuesday. Even as borrowing costs surge, the NY Fed said credit card balances increased by $46 billion last quarter. Over the past year, credit card debt has jumped by $100 billion, or 13%, the biggest percentage increase in more than 20 years. Credit cards typically charge high interest rates when balances aren’t fully paid off, making this an expensive form of debt. The NY Fed said the credit card binge at least partly reflects inflation as prices rise at the fastest pace in more than four decades. “The impacts of inflation are apparent in high volumes of borrowing,” NY Fed researchers wrote in a blog post. High inflation is also making it more expensive to carry a credit card balance because the Federal Reserve is aggressively raising borrowing costs. The Fed raised its benchmark interest rate by three-quarters of a percentage point last week for the second month in a row.  Not only are credit card balances rising, but Americans opened 233 million new credit card accounts during the second quarter, the most since 2008, the NY Fed report found. High inflation is also forcing consumers to dip into their savings. The personal savings rate fell in June to 5.1%, the lowest since August 2009, the Bureau of Labor Statistics said last week. Despite rising debt levels, the NY Fed said consumer balance sheets appear to be in a “strong position” overall. Most of the 2% quarter-over-quarter increase in US household debt to $16.2 trillion was driven by a jump in mortgage borrowing. Student loan balances were little changed at $1.6 trillion. By and large, Americans continued to pay down debt on schedule last quarter, a reflection of the very strong job market. The NY Fed said the share of current debt transitioning into delinquency remains “historically very low,” though it did increase modestly. “Although debt balances are growing rapidly, households in general have weathered the pandemic remarkably well,” the NY Fed said in the report, noting the unprecedented assistance from the federal government during the onset of Covid-19. There are hints, however, that some lower-income and subprime borrowers are now struggling to keep up with their bills. The report found that the delinquency transition rate for credit cards and auto loans is “creeping up,” especially in lower-income areas. “With the supportive policies of the pandemic mostly in the past, there are pockets of borrowers who are beginning to show some distress on their debt,” the report said. Helped by moratoriums and forbearance programs, foreclosures remain “very low,” according to the report. However, credit reports indicate the number of new foreclosures increased by 11,000 during the second quarter, the NY Fed said, potentially signaling the “beginning of a return to more typical levels.”")
    pprint(splits)
    print(len(splits))
