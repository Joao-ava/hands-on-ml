import os
import tarfile
import requests
import pandas as pd
from urllib.request import urlretrieve


DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml3/raw/main"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, "..", "..", "datasets")
IMAGE_PATH = os.path.join(DATASET_DIR, "images")
HOUSING_PATH = os.path.join(DATASET_DIR, "housing")
HOUSING_URL = f"{DOWNLOAD_ROOT}/datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    response = requests.get(housing_url, stream=True)
    content = response.raw.read()
    housing_tgz = tarfile.open(content)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def get_image_path(gitpath: str) -> str:
    if not os.path.isdir(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)

    name = gitpath.split('/')[-1]
    image_path = os.path.join(IMAGE_PATH, name)
    if os.path.exists(image_path):
        return image_path

    urlretrieve(f'{DOWNLOAD_ROOT}/{gitpath}', image_path)
    return image_path


if __name__ == "__main__":
    fetch_housing_data()
