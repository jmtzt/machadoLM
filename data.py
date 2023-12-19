import unicodedata
from tqdm.auto import tqdm
from pathlib import Path


def strip_accents(s):
    """
    Strip accents from a given string.

    This function removes diacritic marks from the input string.

    Args:
    s (str): String from which to remove accents.

    Returns:
    str: The input string with diacritic marks removed.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def simplify(s):
    """
    Simplify a string by stripping accents and removing
    non-alphanumeric characters.

    This function first removes accents using strip_accents,
    then filters the string to include only specific characters
    (alphanumeric and some punctuation).

    Args:
    s (str): The string to be simplified.

    Returns:
    str: The simplified string.
    """
    vocab = (
        "\n 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz:;?!(),-."
    )
    s = "".join(c for c in strip_accents(s) if c in vocab)

    return s


def load_data(path, file_type="txt", verbose=False):
    """
    Load and concatenate text data from .txt files at the specified path.

    Args:
    path (Pathlib.Path): Path to the directory containing .txt files.

    Returns:
    str: Concatenated and simplified text from all .txt files.
    """

    if verbose:
        print(60 * "-")
        print(f"Loading {file_type} files from {path}...")
        print(60 * "-")

    data = ""
    pbar = tqdm(path.glob(f"*.{file_type}"), desc="Processing")
    for file_path in pbar:
        with open(file_path) as f:
            pbar.set_description(f"Processing {file_path.name}")
            data += "\n" + simplify(f.read())

    return data


def preprocess_text(data_path, output_path=None, verbose=False):
    """
    Load, preprocess, and optionally save text data.

    Args:
    data_path (Pathlib.Path): Path to the directory containing text files.
    output_path (Pathlib.Path, optional): Path to save the processed data.
                                          If None, the processed data is
                                          returned and not saved.

    Returns:
    str: Processed text data if output_path is None.
         Otherwise, writes data to file and returns None.
    """
    processed_data = load_data(data_path, verbose=verbose)
    processed_data = simplify(processed_data)

    if verbose:
        print(f"Length of the processed data: {len(processed_data)}")
        print(
            f"First 100 characters of the processed data: {processed_data[:100]}")

    if output_path:
        with open(output_path, "w") as f:
            f.write(processed_data)
            if verbose:
                print(f"Data saved to {output_path}...")
        return None
    else:
        return processed_data


if __name__ == "__main__":
    DATA_PATH = Path("data/raw/txt/romance/")
    OUTPUT_PATH = Path("data/romance.txt")
    preprocess_text(DATA_PATH, OUTPUT_PATH, verbose=True)
