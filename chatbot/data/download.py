import os
import shutil
import sys
import zipfile

import requests

from chatbot import logger

URLS = {
    "cornell_dialogs": "https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip"  # noqa: E501
}


def get_data(name: str, directory: str) -> None:
    """Downloads one of the available datasets.

    Available datasets:
        - `cornel_dialogs`

    Args:
        name (str): Choose one of the available
            datasets.
        directory (str): A local directory path
            to save the data.
    """
    match name.lower():
        case "cornell_dialogs":
            get_cornell_dialogs(directory=directory)
        case _:
            logger.error(f"Unsupported dataset name: `{name}`.")
            raise ValueError(
                f"Unsupported dataset name: `{name}`. "
                f"Choose one of the following: "
                f"{list(URLS.keys())}"
            )


def get_cornell_dialogs(directory: str) -> None:
    """Downloads, and unzips the cornell-dialogs
    data.

    Args:
        directory (str): Local directory to store
            the raw data.
    """
    url = URLS["cornell_dialogs"]
    zipped_filename = f"{os.path.splitext(os.path.basename(url))[0]}.zip"
    zipped_filepath = os.path.join(directory, "raw", zipped_filename)
    download_data(url=url, path=zipped_filepath)
    unzip_data(
        compressed_file=zipped_filepath,
        destination_dir=os.path.join(directory, "raw"),
    )


def download_data(url: str, path: str) -> None:
    """Retrieves data from a source url and saves
    them to a local destination path.

    Args:
        url (str): The url to download the data from.
        path (str): The local destination path to save
            the data.
    """
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    response = requests.get(url=url)

    if response.status_code == 200:
        logger.info(f"Downloading data from {url} to {path}")
        with open(file=path, mode="wb") as f:
            f.write(response.content)
    else:
        logger.error(
            f"Failed to download data. Status code: {response.status_code}"
        )
        sys.exit(1)


def unzip_data(compressed_file: str, destination_dir: str) -> None:
    """Extracts the content of a compressed (`.zip`) file
    to a local directory and removes the compressed file afterward.

    Note:
        If the destination directory and/or its subdirectories
        do not exist, they will be created.

    Args:
        compressed_file (str): The path to the compressed file.
        destination_dir (str): Local directory path to extract the
            content of the compressed file.

    Raises:
        FileNotFoundError: If the compressed file does not exist.
    """
    if not os.path.exists(path=compressed_file):
        raise FileNotFoundError(
            f"The compressed file '{compressed_file}' does not exist."
        )

    if not os.path.isdir(s=destination_dir):
        os.makedirs(destination_dir, exist_ok=True)

    logger.info(f"Unzipping data {compressed_file} to {destination_dir}")
    with zipfile.ZipFile(file=compressed_file, mode="r") as zip_ref:
        zipped_dir_path = os.path.join(destination_dir, zip_ref.namelist()[0])
        if os.path.exists(zipped_dir_path):
            logger.info(f"{zipped_dir_path} already exists. Overwriting...")
            shutil.rmtree(path=zipped_dir_path, ignore_errors=True)
        zip_ref.extractall(path=destination_dir)
    os.remove(path=compressed_file)
