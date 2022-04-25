import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import pandas as pd
import requests
import zipfile
import logging
import base64
import shutil
import time
import ssl
import os


# original version:
# https://github.com/andriygav/MLResearchToolkit


def _filededup(_type, folder):
    r"""
    :return:
    :rtype:
    """
    files = os.listdir(folder)
    list_of_related_files = []
    for file in files:
        tag = file.split(".")[-1]
        if tag == _type:
            list_of_related_files.append(
                (file, os.path.getsize(os.path.join(folder, file)))
            )

    list_of_related_files = sorted(list_of_related_files, key=lambda x: x[1])

    for i in range(len(list_of_related_files) - 1):
        os.remove(os.path.join(folder, list_of_related_files[i][0]))


def _fileclean(folder, _types=[]):
    r"""
    :return:
    :rtype:
    """
    files = os.listdir(folder)
    for file in files:
        tag = file.split(".")[-1]
        if tag not in _types:
            delpath = os.path.join(folder, file)
            if os.path.isdir(delpath):
                shutil.rmtree(delpath)
            else:
                os.remove(delpath)


class UCI(object):
    r"""
    Class for UCI datasets manager system.
    """

    def __init__(
        self,
        enforce=False,
        cache=".uci",
        url="https://archive.ics.uci.edu/ml/datasets.php",
    ):
        r"""
        :param enforce: Rewrite all information in the cache
        :type enforce: bool
        :param cache: Cache directory for meta and dataset saving.
        :type cache: str
        :param url: URL for archive.ics.uci.edu can be changed during the time.
        :type url: str
        """
        self._cache = cache
        self._url = url
        self._enforce = enforce

        self._meta_dir = "metas"
        self._dataset_dir = "datasets"

        self._uci_data_folder = "machine-learning-databases"

        if self._cache and not os.path.exists(self._cache):
            os.makedirs(self._cache)
            os.makedirs(os.path.join(self._cache, self._meta_dir))
            os.makedirs(os.path.join(self._cache, self._dataset_dir))

        if self._cache:
            self._cache = os.path.abspath(".uci")

        self._meta = None

    def get_meta(self, enforce=None):
        r"""
        Get meta infromation about all datasets in the site.

        :param enforce: Enforce rewriting all metadata in
                        the cache or use cached data.
                        Default None: use initial value.
        :type enforce: bool
        """
        if self._meta is not None:
            return self._meta.copy()
        else:
            return self._get_meta(enforce=enforce).copy()

    def _download_meta(self):
        try:
            datasets = pd.read_html(self._url)
        except Exception as e:
            logging.warning("Could not read the table from UCI ML portal, Sorry!")
            raise e

        df = datasets[5]
        nrows = df.shape[0]
        ncols = df.shape[1]

        df = df.iloc[1:]

        df.columns = [
            "Name",
            "Data Types",
            "Task",
            "Feature Types",
            "Samplesize",
            "Featuresize",
            "Year",
        ]

        # добавляем ссылки на выборки
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        uh = urllib.request.urlopen(self._url, context=ctx)
        html = uh.read()

        soup = BeautifulSoup(html, "html.parser")

        urls = dict()
        for tag in soup.find_all("a"):
            if isinstance(tag.contents[0], str):
                urls[tag.contents[0].strip()] = tag.get_attribute_list("href")[0]

        url_lst = []
        relevant_names = df["Name"].values
        for name in relevant_names:
            if name in urls:
                url = "/".join(["/".join(self._url.split("/")[:-1]), urls[name]])
            else:
                url = None
            url_lst.append(url)

        df["URL"] = url_lst

        df.dropna(inplace=True)

        # фиксим мультитаск
        df["Task"] = list(map(lambda x: x.replace(" ", ""), df["Task"].values))
        df["Data Types"] = list(
            map(lambda x: x.replace(" ", ""), df["Data Types"].values)
        )
        df["Feature Types"] = list(
            map(lambda x: x.replace(" ", ""), df["Feature Types"].values)
        )

        df = df.drop_duplicates()
        df.index = list(range(len(df)))
        return df

    def _get_meta(self, enforce=None):
        if enforce is None:
            enforce = self._enforce
        df = None
        cache_file = os.path.join(self._cache, self._meta_dir, "meta.csv")
        if os.path.exists(cache_file) and not enforce:
            df = pd.read_csv(cache_file, sep=";", header=0)
        else:
            df = self._download_meta()
            df.to_csv(cache_file, index=False, sep=";")

        self._meta = df
        return self._meta

    def _download_dataset(self, name):
        ID = base64.b64encode(name.encode("UTF-8")).decode("UTF-8")
        cache_dir = os.path.join(self._cache, self._dataset_dir, ID)

        url = self._meta.loc[self._meta["Name"] == name]["URL"].values[0]
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        uh = urllib.request.urlopen(url, context=ctx)
        html = uh.read()

        soup = BeautifulSoup(html, "html.parser")

        database_id = None
        for tag in soup.find_all("a"):
            sp_href = tag.get_attribute_list("href")[0].split("/")
            if self._uci_data_folder in sp_href:
                database_id = sp_href[sp_href.index(self._uci_data_folder) + 1]

        url = os.path.join(
            self._url,
        )
        url = "/".join(
            ["/".join(self._url.split("/")[:-1]), self._uci_data_folder, database_id]
        )

        uh = urllib.request.urlopen(url, context=ctx)
        html = uh.read().decode()
        soup = BeautifulSoup(html, "html.parser")

        links = []
        for link in soup.find_all("a"):
            links.append(link.attrs["href"])

        links_to_download = []

        if "Index" in links:
            idx = links.index("Index")
        else:
            idx = len(links) - 2
        for i in range(idx + 1, len(links)):
            links_to_download.append(os.path.join(url, str(links[i])))

        try:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            for link in links_to_download:
                if not os.path.basename(link):
                    continue

                filename = os.path.join(cache_dir, os.path.basename(link))
                r = requests.get(link, stream=True)

                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)

                _type = os.path.basename(filename).split(".")[-1]
                if _type == "zip":
                    with zipfile.ZipFile(filename, "r") as zip_ref:
                        zip_ref.extractall(cache_dir)
                    os.remove(filename)
                _filededup("csv", cache_dir)
                _filededup("data", cache_dir)
                _filededup("train", cache_dir)
                _fileclean(cache_dir, _types=["csv", "data", "train", "names"])

        except Exception as e:
            shutil.rmtree(cache_dir)
            logging.warning(
                f"Something wrong with dataset `{name}` downloading {str(e)}"
            )

    def _get_dataset(self, name, enforce=None, download=True):
        if enforce is None:
            enforce = self._enforce
        ID = base64.b64encode(name.encode("UTF-8")).decode("UTF-8")

        dataset = dict()
        temp = self._meta.loc[self._meta["Name"] == name].to_dict().items()
        dataset["meta"] = {key: list(item.values())[0] for key, item in temp}
        dataset["data"] = None

        cache_dir = os.path.join(self._cache, self._dataset_dir, ID)
        if not (os.path.exists(cache_dir) and not enforce) and download:
            self._download_dataset(name)

        try:
            files = os.listdir(cache_dir)
            tags = list(map(lambda x: x.split(".")[-1], files))

            if "data" in tags:
                dataset["data"] = pd.read_csv(
                    os.path.join(cache_dir, files[tags.index("data")])
                )

                if "names" in tags:
                    with open(os.path.join(cache_dir, files[tags.index("names")])) as f:
                        dataset["names"] = f.read()
            elif "csv" in tags:
                dataset["data"] = pd.read_csv(
                    os.path.join(cache_dir, files[tags.index("csv")])
                )
        except Exception as e:
            logging.warning(f"Something wrong with dataset `{name}` reading {str(e)}")

        return dataset

    def get_dataset(self, name, enforce=None, download=True):
        r"""
        Get dataset by given name.

        :param name: Name of the dataset for downloading.
        :type name: str
        :param enforce: Enforce rewriting all metadata in
                        the cache or use cached data.
                        Default None: use initial value.
        :type enforce: bool
        :param download: Download dataset or not.
                         If False and dataset is not presented than return None.
        :type download: bool
        """
        if self._meta is None:
            self._get_meta()
        return self._get_dataset(name, enforce=enforce, download=download)

    def __len__(self):
        return len(self._meta)

    def __iter__(self):
        def generator():
            for name in datasets._meta["Name"].values:
                yield self._get_dataset(name)

        return generator()
