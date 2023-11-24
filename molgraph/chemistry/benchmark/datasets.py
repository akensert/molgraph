import numpy as np
import pandas as pd

import os
import io
import gzip
import zipfile
import tarfile
from urllib import request

from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import Tuple
from typing import List
from typing import Callable
from typing import Dict
from typing import Any

from molgraph.chemistry.benchmark import configs
from molgraph.chemistry.benchmark import splitters



class Dataset(dict):

    '''A dataset.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data = self.copy()
        self.clear()

        if isinstance(data.get('indices'), dict):
            for set_name, set_indices in data['indices'].items():

                new_data = {
                    'x': data['x'][set_indices],
                    'y': data['y'][set_indices],
                    'index': set_indices
                }

                if data.get('y_mask') is not None:
                    new_data['y_mask'] = data['y_mask'][set_indices]

                self[set_name] = Dataset(new_data)
        else:
            self['x'] = data['x']
            self['y'] = data['y']
            if data.get('y_mask') is not None:
                self['y_mask'] = data['y_mask']

            if data.get('indices') is None:
                data['indices'] = np.arange(len(data['x']))

            self['index'] = data['indices']

    def __repr__(self):
        if any(isinstance(v, self.__class__) for v in self.values()):
            fields = ', '.join([
                f"'{k}': {v}" for (k, v) in self.items()
            ])
            return '<{class_name}: {{{fields}}}>'.format(
                class_name=self.__class__.__name__.lstrip('_'),
                fields=fields
            )
        fields = []
        for k, v in self.items():
            if not hasattr(v, 'shape') and not hasattr(v, 'dtype'):
                fields.append(f"'{k}': {v}")
            else:
                fields.append(f"'{k}': array(shape={v.shape}, dtype={v.dtype})")

        return "<{class_name}: {{{fields}}}>".format(
            class_name=self.__class__.__name__.lstrip('_'),
            fields=', '.join(fields)
        )


class DatasetFactory:

    '''A dataset factory.

    Obtain a dataset with ease:

    >>> esol_dataset = molgraph.chemistry.datasets.get('esol') # doctest: +SKIP

    List available (registered) datasets:

    >>> molgraph.chemistry.datasets.registered_datasets # doctest: +SKIP
    ['smrt',
     'qm7',
     'qm8',
     'qm9',
     'pcba',
     'muv',
     'hiv',
     'bace',
     'bbbp',
     'tox21',
     'toxcast',
     'sider',
     'clintox',
     'esol',
     'freesolv',
     'lipophilicity']
    '''

    def __init__(self) -> None:
        self._classes = {}
        self._configs = {}
        self._names = []

    def register_loader(
        self,
        loader: 'DatasetLoader',
        name: str,
        config: Dict[str, Any],
        overwrite: bool = False
    ) -> None:
        '''Registers a derived class of ``DatasetLoader``.

        After registration, the dataset, corresponding to the ``DatasetLoader``,
        can be obtained.
        '''
        for field in [
            'file_name', 'url', 'molecule_column', 'label_columns'
        ]:
            if field not in config:
                raise ValueError(
                    f"Could not register `{loader.__name__}`, " +
                    f"`{field}` not found in `config`."
                )

        if name in self._classes and not overwrite:
            raise ValueError(f"`{name}` is already registered")

        self._classes[name] = loader
        self._configs[name] = config
        self._names.append(name)

    def register(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> Callable[['DatasetLoader'], 'DatasetLoader']:
        '''Registers a derived class of ``DatasetLoader`` via decoration.

        After registration, the dataset, corresponding to the ``DatasetLoader``,
        can be obtained.
        '''
        def wrapper(loader: 'DatasetLoader') -> 'DatasetLoader':
            self.register_loader(loader, name, config)
            return loader
        return wrapper

    def get(self, name: str, **config) -> 'Dataset':
        '''Get dataset, by name.

        Args:
            name (str):
                Name of the dataset (see ``registered_datasets``).
            **config
                Overwrite default configuration by specifying (as keyword
                arugments) new configuration. E.g.,
                ``dataset.get('esol', splitter=None)`` to avoid splitting
                the dataset.

        Returns:
            A ``Dataset`` or nested ``Dataset``.
        '''
        cls = self._classes.get(name)
        default_config = dict(self._configs.get(name))
        if cls is None:
            raise ValueError(f"{name} could not be found.")

        default_config.update(config)

        dataset_loader = cls(**default_config)
        return dataset_loader.load_dataset()

    def get_config(self, name: str) -> Dict[str, Any]:
        '''Get config of dataset, by name.

        Args:
            name (str):
                Name of the dataset (see ``registered_datasets``).

        Returns:
            The configuration (dict) of the specified dataset.
        '''
        return self._configs.get(name)

    @property
    def registered_datasets(self) -> List[str]:
        'Lists all registered datasets.'
        return list(self._classes.keys())

    def __repr__(self) -> str:
        datasets = list(self._classes.keys())
        return f'DatasetFactory(registered_datasets={datasets})'

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self._classes):
            name = self._names[self.index]
            self.index += 1
            return name, self.get(name)
        raise StopIteration


datasets = DatasetFactory()


class DatasetLoader(ABC):

    '''Base class for dataset loaders.

    A dataset loader loads a `Dataset` via the `load_dataset` method. For this
    to work, the child class should define a `download_file` and a `read_file`
    method.
    '''

    def __init__(
        self,
        url: str,
        file_name: str,
        molecule_column: str,
        label_columns: List[str],
        cache_dir: Optional[str] = None,
        label_masking: Optional[bool] = None,
        splitter: Optional[Any] = None,
        **kwargs
    ) -> None:

        self.url = url
        self.file_name = file_name
        self.molecule_column = molecule_column
        self.label_columns = label_columns
        self.cache_dir = cache_dir
        self.label_masking = label_masking
        self.splitter = splitter

    @abstractmethod
    def download_file(self) -> None:
        '''Downloads file to `self.file_path`. This file is then to be
        read by the user defined `read_file` method.
        '''
        pass

    @abstractmethod
    def read_file(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Reads (downloaded) file from `self.file_path`, returning its
        molecules (`x`) and labels (`y`). Returned molecules should be
        SMILES, InChI, SDF block (represented as a Python string) or RDKit
        molecule objects.
        '''
        pass

    def load_dataset(self) -> 'Dataset':
        '''Loads a `Dataset` with the help of `download_file` and `read_file`
        methods.
        '''
        self.file_path = _get_file_path(
            self.file_name, self.cache_dir)

        if not os.path.exists(self.file_path):
            self.download_file()

        x, y = self.read_file()

        data = {'x': x, 'y': y}

        if getattr(self, 'label_masking', False):
            data['y_mask'] = np.isfinite(data['y']).astype(data['y'].dtype)
            data['y'] = np.nan_to_num(data['y'], nan=0.0)

        splitter = splitters.deserialize(getattr(self, 'splitter', None))

        if splitter is not None:
            indices = splitter.split(x, y)
            data['indices'] = {'train': indices[0]}
            if len(indices) == 2:
                data['indices']['test'] = indices[1]
            else:
                data['indices']['validation'] = indices[1]
                data['indices']['test'] = indices[2]

        return Dataset(data)


class MoleculeNetDatasetLoader(DatasetLoader):

    def download_file(self):
        '''Downloads file to `self.file_path`. This file is then to be
        read by the user defined `read_file` method.
        '''
        if self.url.endswith('.tar.gz'):
            _download_tar_file(self.url, self.file_path)
        elif self.url.endswith('.gz'):
            _download_gzip_file(self.url, self.file_path)
        else:
            _download_file(self.url, self.file_path)

    def read_file(self, processes=None):
        '''Reads (downloaded) file from `self.file_path`, returning its
        molecules (`x`) and labels (`y`). Returned molecules should be
        SMILES, InChI, SDF block (represented as a Python string) or RDKit
        molecule objects.
        '''
        df = pd.read_csv(self.file_path)
        if self.file_name == 'toxcast_data.csv':
            df = df[df[self.molecule_column] != 'FAIL']
        labels = df[self.label_columns].to_numpy('float32')
        molecules = df[self.molecule_column].to_numpy()
        return molecules, labels


@datasets.register(name='smrt', config=configs.smrt)
class SMRTDatasetLoader(DatasetLoader):

    def download_file(self):
        _download_file(self.url, self.file_path)

    def read_file(self, processes=None):
        df = pd.read_csv(self.file_path, sep=';')
        df = df[df[self.label_columns[0]] >= 300]
        molecules = df[self.molecule_column].to_numpy()
        labels = df[self.label_columns].to_numpy('float32') / 60.
        return molecules, labels


@datasets.register(name='qm7', config=configs.qm7)
class QM7DatasetLoader(MoleculeNetDatasetLoader):

    def read_file(self, processes=None):
        with open(self.file_path) as fh:
            molecules = fh.read()
            molecules = molecules.split('\n$$$$\n')
            molecules = molecules[:-1]
        molecules = np.array([x for x in molecules if x.startswith('gdb')])
        df = pd.read_csv(self.file_path + '.csv')
        labels = df[self.label_columns].to_numpy('float32')
        return molecules, labels


@datasets.register(name='qm8', config=configs.qm8)
class QM8DatasetLoader(QM7DatasetLoader):
    pass


@datasets.register(name='qm9', config=configs.qm9)
class QM9DatasetLoader(QM7DatasetLoader):
    pass


@datasets.register(name='pcba', config=configs.pcba)
class PCBADatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='muv', config=configs.muv)
class MUVDatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='hiv', config=configs.hiv)
class HIVDatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='bace', config=configs.bace)
class BaceDatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='bbbp', config=configs.bbbp)
class BBBPDatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='tox21', config=configs.tox21)
class Tox21DatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='toxcast', config=configs.toxcast)
class ToxCastDatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='sider', config=configs.sider)
class SiderDatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='clintox', config=configs.clintox)
class ClinToxDatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='esol', config=configs.esol)
class ESOLDatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='freesolv', config=configs.freesolv)
class FreeSolvDatasetLoader(MoleculeNetDatasetLoader):
    pass


@datasets.register(name='lipophilicity', config=configs.lipophilicity)
class LipophilicityDatasetLoader(MoleculeNetDatasetLoader):
    pass


def _download_file(
    url: str,
    save_path: str
) -> None:
    with request.urlopen(url) as http_response:
        with open(save_path, 'w') as fh:
            fh.write(http_response.read().decode('utf-8'))

def _download_zip_file(
    url: str,
    save_path: str
) -> None:
    with request.urlopen(url) as http_response:
        with zipfile.ZipFile(io.BytesIO(http_response.read())) as zip_file:
            zip_file.extractall(os.path.dirname(save_path))

def _download_gzip_file(
    url: str,
    save_path: str
) -> None:
    with request.urlopen(url) as http_response:
        with gzip.GzipFile(fileobj=http_response) as gzip_file:
            with open(save_path, 'wb') as f:
                f.write(gzip_file.read())

def _download_tar_file(
    url: str,
    save_path: str
) -> None:
    with request.urlopen(url) as http_response:
        with tarfile.open(fileobj=io.BytesIO(http_response.read())) as tar_file:
            tar_file.extractall(os.path.dirname(save_path))

def _get_file_path(
    fname: str,
    cache_dir: Optional[str] = None
) -> str:
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.molgraph')
        os.makedirs(cache_dir, exist_ok=True)
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.molgraph')
    datadir = os.path.join(datadir_base, 'datasets')
    os.makedirs(datadir, exist_ok=True)
    return os.path.join(datadir, fname)
