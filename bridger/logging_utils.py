"""A set of utility functions related to logging."""

import pathlib

def create_serialization_dir(dirname: str) -> None:
    """Creates directory dirname if it doesn't exist.

    Clears the contents of the directory if the dirname existed previously.
    
    Args:
      dirname: The name of the directory to create.
    """
    path = pathlib.Path(dirname)
    path.mkdir(parents=True, exist_ok=True)
    for filepath in path.iterdir():
        filepath.unlink()
        
