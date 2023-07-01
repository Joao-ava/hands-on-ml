import os
import tarfile
import requests
import pandas as pd
from pathlib import Path
from email import policy
from email.parser import BytesParser
from urllib.request import urlretrieve


DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml3/raw/main"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, "..", "..", "datasets")
IMAGE_PATH = os.path.join(DATASET_DIR, "images")
HOUSING_PATH = os.path.join(DATASET_DIR, "housing")
TITANIC_PATH = os.path.join(DATASET_DIR, "titanic")
SPAM_PATH = os.path.join(DATASET_DIR, "spam")
HOUSING_URL = f"{DOWNLOAD_ROOT}/datasets/housing/housing.tgz"
TITANIC_URL = "https://github.com/ageron/data/raw/main/titanic.tgz"
SPAM_BASE_URL = "http://spamassassin.apache.org/old/publiccorpus"
SPAM_URL = f"{SPAM_BASE_URL}/20030228_spam.tar.bz2"
HAM_URL = f"{SPAM_BASE_URL}/20030228_easy_ham.tar.bz2"


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


def load_titanic_data():
    if not os.path.isdir(TITANIC_PATH):
        os.makedirs(TITANIC_PATH)

    titanic_train_filepath = os.path.join(TITANIC_PATH, 'train.csv')
    titanic_test_filepath = os.path.join(TITANIC_PATH, 'test.csv')
    if not os.path.isfile(titanic_train_filepath):
        tarball_path = os.path.join(TITANIC_PATH, 'titanic.tgz')
        urlretrieve(TITANIC_URL, tarball_path)
        with tarfile.open(tarball_path) as f:
            f.extractall(DATASET_DIR)

    return pd.read_csv(titanic_train_filepath), pd.read_csv(titanic_test_filepath)


def load_email(filepath):
    with open(filepath, 'rb') as f:
        content = BytesParser(policy=policy.default).parse(f)

    return content


def replace_default_charset(file):
    """Resolve problem of windows opens file don't know DEFAULT_CHARSET"""
    with open(file, 'rb') as f:
        content = f.read()\
            .replace(b'charset="DEFAULT_CHARSET"', b'charset="utf-8"')\
            .replace(b'charset=unknown-8bit', b'charset="utf-8"')\
            .replace(b'charset="DEFAULT"', b'charset="utf-8"')

    with open(file, 'wb') as f:
        f.write(content)


def load_spam_data():
    spam_dir = os.path.join(SPAM_PATH, 'spam')
    ham_dir = os.path.join(SPAM_PATH, 'easy_ham')
    path_urls = (
        (spam_dir, SPAM_URL),
        (ham_dir, HAM_URL)
    )
    for file_dir, url in path_urls:
        if os.path.isdir(file_dir):
            continue

        os.makedirs(file_dir)
        path = os.path.join(file_dir, 'data.tar.bz2')
        urlretrieve(url, path)
        with tarfile.open(path) as f:
            f.extractall(SPAM_PATH)

        os.remove(path)

    exclude_files = ['cmds']
    spam_filenames = [
        file for file in sorted(Path(spam_dir).iterdir())
        if len(file.name) > 20 and file.name not in exclude_files
    ]
    for file in spam_filenames:
        replace_default_charset(file)

    spam_emails = [load_email(filepath) for filepath in spam_filenames]
    ham_filenames = [
        file for file in sorted(Path(ham_dir).iterdir())
        if len(file.name) > 20 and file.name not in exclude_files
    ]
    for file in ham_filenames:
        replace_default_charset(file)

    ham_emails = [load_email(filepath) for filepath in ham_filenames]
    return spam_emails, ham_emails


def get_image_path(git_path: str) -> str:
    if not os.path.isdir(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)

    name = git_path.split('/')[-1]
    image_path = os.path.join(IMAGE_PATH, name)
    if os.path.exists(image_path):
        return image_path

    urlretrieve(f'{DOWNLOAD_ROOT}/{git_path}', image_path)
    return image_path


if __name__ == "__main__":
    load_housing_data()
    load_titanic_data()
    load_spam_data()
