from .storage import BinaryStorage, DataFramesStorage, JsonStorage, TextStorage

dataframes = DataFramesStorage()
json = JsonStorage()
text = TextStorage()
binary = BinaryStorage()

__all__ = [
    "dataframes",
    "json",
    "binary",
    "text",
]
