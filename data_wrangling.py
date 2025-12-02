import pandas as pd
from typing import Union, List, Dict, Any

def dataframe_to_dict(df: pd.DataFrame, keys: Union[str, List[str]], value: str) -> Dict[Any, Any]:
    if isinstance(keys, list):  # If `keys` is a list of columns
        # Use tuples as dictionary keys
        keys_as_tuples = df[keys].apply(tuple, axis=1)
        return dict(zip(keys_as_tuples, df[value]))
    else:  # If `keys` is a single column
        # Use a single column as dictionary keys
        return dict(zip(df[keys], df[value]))

