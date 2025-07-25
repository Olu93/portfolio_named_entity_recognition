import pandas as pd
from utils.typings import TextInput
import logging
import time
import numpy as np

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


def convert_y_to_list(y: TextInput):
    start_time = time.time()
    final_result = []
    if isinstance(y, str):
        final_result.extend([y])
    elif isinstance(y, pd.Series):
        # TODO: Cover cases in which the series elements are strings and not list
        final_result.extend(y.apply(lambda x: x.split(';')).tolist())
    elif isinstance(y, pd.DataFrame):
        # TODO: Cover cases in which the dataframe elements are strings and not list
        final_result.extend(y.apply(lambda x: x.split(';')).values.tolist())
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


# Test cases if run as main
if __name__ == "__main__":
    # TODO: Convert to pytest tests
    # Single example case
    X = "Hello, I am John Doe"
    y = ["John Doe", "Jane Doe"]
    print(convert_X_to_list(X))
    print(convert_y_to_list(y))

    # Multi example case
    X = ["Hello, I am John Doe", "Hello, I am Jane Doe"]
    y = [["John Doe", "Jane Doe"], ["John Doe", "Jane Doe"]]
    print(convert_X_to_list(X))
    print(convert_y_to_list(y))
