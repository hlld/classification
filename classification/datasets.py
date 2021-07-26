import lzma
import os
import hashlib
import tarfile
import urllib
import zipfile
import codecs
import gzip
import numpy as np
import scipy
import torch
from tqdm import tqdm


class _BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_augment,
                 hyp_params):
        self.data_augment = data_augment
        self.hyp_params = hyp_params

    @staticmethod
    def gen_bar_updater():
        pbar = tqdm(total=None)

        def bar_update(count, block_size, total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

    @staticmethod
    def check_md5(fpath, target, chunk_size=1024 * 1024):
        md5 = hashlib.md5()
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
        return target == md5.hexdigest()

    def check_integrity(self, fpath, md5=None):
        if not os.path.isfile(fpath):
            return False
        if md5 is None:
            return True
        return self.check_md5(fpath, md5)

    def download_url(self, url, root, filename=None, md5=None):
        root = os.path.expanduser(root)
        if not filename:
            filename = os.path.basename(url)
        fpath = os.path.join(root, filename)

        os.makedirs(root, exist_ok=True)
        # Check if file is already present locally
        if self.check_integrity(fpath, md5):
            print('Using downloaded and verified file ' + fpath)
        else:
            try:
                print('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url,
                                           fpath,
                                           reporthook=self.gen_bar_updater())
            except (urllib.error.URLError, IOError) as err:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead,'
                          ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(
                        url,
                        fpath,
                        reporthook=self.gen_bar_updater())
                else:
                    raise err
            # Check integrity of downloaded file
            if not self.check_integrity(fpath, md5):
                raise RuntimeError("File not found or corrupted")

    @staticmethod
    def extract_archive(from_path,
                        to_path=None,
                        remove_finished=False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        if from_path.endswith(".tar"):
            with tarfile.open(from_path, 'r') as tar:
                tar.extractall(path=to_path)
        elif from_path.endswith(".tar.gz") or from_path.endswith(".tgz"):
            with tarfile.open(from_path, 'r:gz') as tar:
                tar.extractall(path=to_path)
        elif from_path.endswith(".tar.xz"):
            with tarfile.open(from_path, 'r:xz') as tar:
                tar.extractall(path=to_path)
        elif from_path.endswith(".gz") and not from_path.endswith(".tar.gz"):
            to_path = os.path.join(to_path, os.path.splitext(
                os.path.basename(from_path))[0])
            with open(to_path, "wb") as out_f, \
                    gzip.GzipFile(from_path) as zip_f:
                out_f.write(zip_f.read())
        elif from_path.endswith(".zip"):
            with zipfile.ZipFile(from_path, 'r') as z:
                z.extractall(to_path)
        else:
            raise ValueError('Extraction of %s not supported' % from_path)
        if remove_finished:
            os.remove(from_path)

    def download_and_extract_archive(self,
                                     url,
                                     download_root,
                                     extract_root=None,
                                     filename=None,
                                     md5=None,
                                     remove_finished=False):
        download_root = os.path.expanduser(download_root)
        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        self.download_url(url, download_root, filename, md5)
        archive = os.path.join(download_root, filename)
        print("Extracting %s to %s" % (archive, extract_root))
        self.extract_archive(archive, extract_root, remove_finished)

    def data_length(self):
        raise NotImplementedError

    def get_image(self, index):
        raise NotImplementedError

    def get_target(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        image = self.get_image(index)
        target = self.get_target(index)
        return image, target

    def __len__(self):
        return self.data_length()


class MNIST(_BaseDataset):
    def __init__(self,
                 data_root,
                 data_split,
                 data_augment=False,
                 hyp_params=None,
                 download=True):
        super(MNIST, self).__init__(data_augment, hyp_params)
        assert data_split in ['train', 'test']
        self.data_root = data_root
        self.data_split = data_split
        if download:
            self.download()
        self.raw_path = os.path.join(data_root,
                                     'MNIST',
                                     'raw')
        self.processed_path = os.path.join(data_root,
                                           'MNIST',
                                           'processed')
        self.train_file = 'training.pt'
        self.test_file = 'test.pt'
        if data_split == 'train':
            data_file = self.train_file
        else:
            data_file = self.test_file
        self.images, self.targets = torch.load(
                os.path.join(self.processed_path, data_file))

    def download(self):
        resources = [
            ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
             "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
             "d53e105ee54ea40749a09fcbcd1e9432"),
            ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
             "9fb629c4189551a2d022fa330f9573f3"),
            ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
             "ec29112dd5afa0611ce80d1b7f02629c")
        ]
        train_path = os.path.join(self.processed_path, self.train_file)
        test_path = os.path.join(self.processed_path, self.test_file)
        if not os.path.exists(train_path) or \
                not os.path.exists(test_path):
            os.makedirs(self.raw_path, exist_ok=True)
            os.makedirs(self.processed_path, exist_ok=True)

            for url, md5 in resources:
                filename = url.rpartition('/')[2]
                self.download_and_extract_archive(url,
                                                  download_root=self.raw_path,
                                                  filename=filename,
                                                  md5=md5)
            training_set = (self.read_image_file(
                os.path.join(self.raw_path, 'train-images-idx3-ubyte')),
                            self.read_label_file(
                os.path.join(self.raw_path, 'train-labels-idx1-ubyte')))
            test_set = (self.read_image_file(
                os.path.join(self.raw_path, 't10k-images-idx3-ubyte')),
                        self.read_label_file(
                os.path.join(self.raw_path, 't10k-labels-idx1-ubyte')))
            with open(os.path.join(self.processed_path,
                                   self.train_file), 'wb') as fd:
                torch.save(training_set, fd)
            with open(os.path.join(self.processed_path,
                                   self.test_file), 'wb') as fd:
                torch.save(test_set, fd)
            print('Download Done')

    def read_label_file(self, path):
        with open(path, 'rb') as fd:
            x = self.read_sn3_pascalvincent_tensor(fd, strict=False)
        assert x.dtype == torch.uint8
        assert x.ndimension() == 1
        return x.long()

    def read_image_file(self, path):
        with open(path, 'rb') as fd:
            x = self.read_sn3_pascalvincent_tensor(fd, strict=False)
        assert x.dtype == torch.uint8
        assert x.ndimension() == 3
        return x

    def read_sn3_pascalvincent_tensor(self, path, strict=True):
        if not hasattr(self.read_sn3_pascalvincent_tensor, 'typemap'):
            self.read_sn3_pascalvincent_tensor.typemap = {
                8: (torch.uint8, np.uint8, np.uint8),
                9: (torch.int8, np.int8, np.int8),
                11: (torch.int16, np.dtype('>i2'), 'i2'),
                12: (torch.int32, np.dtype('>i4'), 'i4'),
                13: (torch.float32, np.dtype('>f4'), 'f4'),
                14: (torch.float64, np.dtype('>f8'), 'f8')}

        with self.open_maybe_compressed_file(path) as f:
            data = f.read()

        magic = int(codecs.encode(data[0:4], 'hex'), 16)
        nd = magic % 256
        ty = magic // 256
        assert 1 <= nd <= 3
        assert 8 <= ty <= 14
        m = self.read_sn3_pascalvincent_tensor.typemap[ty]
        s = []
        for k in range(nd):
            s.append(int(codecs.encode(data[4 * (k + 1): 4 * (k + 2)],
                                       'hex'), 16))
        parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
        assert parsed.shape[0] == np.prod(s) or not strict
        return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

    @staticmethod
    def open_maybe_compressed_file(path):
        if not isinstance(path, torch._six.string_classes):
            return path
        if path.endswith('.gz'):
            return gzip.open(path, 'rb')
        if path.endswith('.xz'):
            return lzma.open(path, 'rb')
        return open(path, 'rb')

    def data_length(self):
        return len(self.images)

    def get_image(self, index):
        return self.images[index]

    def get_target(self, index):
        return int(self.targets[index])


class SVHN(_BaseDataset):
    def __init__(self,
                 data_root,
                 data_split,
                 data_augment=False,
                 hyp_params=None,
                 download=True):
        super(SVHN, self).__init__(data_augment, hyp_params)
        assert data_split in ['train', 'test', 'extra']
        resources = {
            'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                      "train_32x32.mat",
                      "e26dedcc434d2e4c54c9b2d4a06d8373"],
            'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                     "test_32x32.mat",
                     "eb5a983be6a315427106f1b164d9cef3"],
            'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                      "extra_32x32.mat",
                      "a93ce644f1a588dc4d68dda5feec44a7"]
        }
        self.data_root = data_root
        self.data_split = data_split
        file_url = resources[data_split][0]
        file_name = resources[data_split][1]
        file_md5 = resources[data_split][2]
        file_path = os.path.join(self.data_root, file_name)
        if download and not self.check_integrity(file_path, file_md5):
            self.download_url(file_url,
                              self.data_root,
                              file_name,
                              file_md5)
            print('Download Done')
        loaded_mat = scipy.io.loadmat(os.path.join(data_root, file_name))
        self.images = np.transpose(loaded_mat['X'], (3, 2, 0, 1))
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat['y'].astype(np.int64).squeeze()
        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)

    def data_length(self):
        return len(self.images)

    def get_image(self, index):
        return self.images[index]

    def get_target(self, index):
        return int(self.targets[index])
