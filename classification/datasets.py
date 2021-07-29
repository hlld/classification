import argparse
import math
import os
import random
import numpy as np
import cv2
import scipy.io as sio
import pickle
import codecs
import tarfile
import torch
from tqdm import tqdm
from copy import deepcopy
from classification.tools import torch_zero_rank_first


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 data_root,
                 data_type,
                 data_split,
                 input_size,
                 batch_size=16,
                 data_augment=False,
                 hyp_params=None,
                 download=True,
                 shuffle=False,
                 num_workers=0,
                 local_rank=-1):
        with torch_zero_rank_first(local_rank):
            if hyp_params is not None:
                hyp_params = deepcopy(hyp_params)
            if data_type in ['ilsvrc2012', 'custom']:
                dataset = ImageFolder(data_root,
                                      data_split,
                                      input_size=input_size,
                                      data_augment=data_augment,
                                      hyp_params=hyp_params)
            elif data_type in ['mnist', 'svhn', 'cifar10', 'cifar100']:
                if data_type == 'mnist':
                    dataset_builder = MNIST
                elif data_type == 'svhn':
                    dataset_builder = SVHN
                elif data_type == 'cifar10':
                    dataset_builder = CIFAR10
                else:
                    dataset_builder = CIFAR100
                dataset = dataset_builder(data_root,
                                          data_split,
                                          input_size=input_size,
                                          data_augment=data_augment,
                                          hyp_params=hyp_params,
                                          download=download)
            else:
                raise ValueError('Unknown type %s' % data_type)
        batch_size = min(batch_size, len(dataset))
        num_workers = min([os.cpu_count(),
                           batch_size,
                           num_workers])
        if batch_size == 1:
            num_workers = 0
        sampler = None
        if local_rank != -1:
            shuffle = False
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        super(DataLoader, self).__init__(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         sampler=sampler)

    @staticmethod
    def default_params(data_type):
        input_size, in_channels, num_classes, image_mean, image_std = \
            -1, -1, -1, [], []
        if data_type == 'mnist':
            input_size = 28
            in_channels = 1
            num_classes = 10
            image_mean = [0.5]
            image_std = [0.5]
        elif data_type == 'svhn':
            input_size = 32
            in_channels = 3
            num_classes = 10
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        elif data_type == 'cifar10':
            input_size = 32
            in_channels = 3
            num_classes = 10
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        elif data_type == 'cifar100':
            input_size = 32
            in_channels = 3
            num_classes = 100
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        elif data_type == 'ilsvrc2012':
            input_size = 224
            in_channels = 3
            num_classes = 1000
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        return input_size, in_channels, num_classes, image_mean, image_std


class _BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 input_size,
                 data_augment=False,
                 hyp_params=None):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if hyp_params is None:
            hyp_params = {'flip': 0.5,
                          'crop': 0.5,
                          'mean': [0.5, 0.5, 0.5],
                          'std': [0.5, 0.5, 0.5]}
        self.input_size = input_size
        self.data_augment = data_augment
        self.hyp_params = hyp_params
        self.classes = []

    @staticmethod
    def gen_bar_updater():
        pbar = tqdm(total=None)

        def bar_update(count,
                       block_size,
                       total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

    @staticmethod
    def check_md5(fpath,
                  target,
                  chunk_size=1024 * 1024):
        import hashlib
        md5 = hashlib.md5()
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
        return target == md5.hexdigest()

    def check_integrity(self,
                        fpath,
                        md5=None):
        if not os.path.isfile(fpath):
            return False
        if md5 is None:
            return True
        return self.check_md5(fpath, md5)

    def download_url(self,
                     url,
                     root,
                     filename=None,
                     md5=None):
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
                import urllib
                print('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url,
                                           fpath,
                                           reporthook=self.gen_bar_updater())
            except (urllib.error.URLError, IOError) as err:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https instead of http,'
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
            import gzip
            to_path = os.path.join(to_path, os.path.splitext(
                os.path.basename(from_path))[0])
            with open(to_path, "wb") as out_f, \
                    gzip.GzipFile(from_path) as zip_f:
                out_f.write(zip_f.read())
        elif from_path.endswith(".zip"):
            import zipfile
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

        self.download_url(url,
                          download_root,
                          filename,
                          md5)
        archive = os.path.join(download_root, filename)
        print("Extracting %s to %s" % (archive, extract_root))
        self.extract_archive(archive,
                             extract_root,
                             remove_finished)

    def data_length(self):
        raise NotImplementedError

    def get_image(self, index):
        raise NotImplementedError

    def get_target(self, index):
        raise NotImplementedError

    @staticmethod
    def random_size_rect(image,
                         scale=(0.08, 1.0),
                         ratio=(3 / 4.0, 4 / 3.0),
                         max_times=10):
        height, width = image.shape[:2]
        area = height * width
        for _ in range(max_times):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]),
                         math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                return top, left, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        top = (height - h) // 2
        left = (width - w) // 2
        return top, left, h, w

    def fixed_size_rect(self, image):
        height, width = image.shape[:2]
        th, tw = self.input_size
        if height == th and width == tw:
            return 0, 0, height, width

        top = random.randint(0, height - th)
        left = random.randint(0, width - tw)
        return top, left, th, tw

    def random_crop(self,
                    image,
                    inplace=False):
        if not inplace:
            image = deepcopy(image)
        height, width = image.shape[:2]
        input_h, input_w = self.input_size
        h_thresh, w_thresh = (round(1.25 * input_h),
                              round(1.25 * input_w))
        padding = None
        if height < h_thresh or width < w_thresh:
            padding_h, padding_w = (0, 0), (0, 0)
            if height < h_thresh:
                padding_size = h_thresh - height
                padding_h = (padding_size // 2,
                             padding_size // 2 + padding_size % 2)
            if width < w_thresh:
                padding_size = w_thresh - width
                padding_w = (padding_size // 2,
                             padding_size // 2 + padding_size % 2)
            padding = (padding_h, padding_w, (0, 0))
        if padding is not None:
            image = np.pad(image,
                           padding,
                           mode='constant',
                           constant_values=0)
            top, left, h, w = self.fixed_size_rect(image)
        else:
            top, left, h, w = self.random_size_rect(image)
        return image[top:(top + h), left:(left + w), :]

    def normalize(self,
                  image,
                  inplace=False):
        if not inplace:
            image = deepcopy(image)
        mean = np.array(self.hyp_params['mean'], dtype=np.float32)
        std = np.array(self.hyp_params['std'], dtype=np.float32)
        if (std == 0).any():
            raise ValueError('Normalization division by zero')
        image /= 255.0
        image -= mean
        image /= std
        return image

    def __getitem__(self, index):
        image = self.get_image(index)
        target = self.get_target(index)
        image = image.astype(np.float32)
        if self.data_augment:
            if random.random() < self.hyp_params['flip']:
                image = np.fliplr(image)
            if random.random() < self.hyp_params['crop']:
                image = self.random_crop(image)
        if image.shape[:2] != self.input_size:
            interpolation = cv2.INTER_LINEAR
            if self.data_augment:
                interpolation = cv2.INTER_AREA
            image = cv2.resize(image,
                               (self.input_size[1], self.input_size[0]),
                               interpolation=interpolation)
            if len(image.shape) == 2:
                image = image[:, :, None]
        image = self.normalize(image, inplace=True)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image)
        # Convert to [C, H, W] format
        image = image.permute((2, 0, 1))
        return image, target

    def __len__(self):
        return self.data_length()


class MNIST(_BaseDataset):
    def __init__(self,
                 data_root,
                 data_split,
                 input_size,
                 data_augment=False,
                 hyp_params=None,
                 download=True):
        super(MNIST, self).__init__(input_size=input_size,
                                    data_augment=data_augment,
                                    hyp_params=hyp_params)
        assert data_split in ['train', 'val', 'test']
        mean, std = (self.hyp_params['mean'][0],
                     self.hyp_params['std'][0])
        self.hyp_params['mean'] = mean
        self.hyp_params['std'] = std
        self.data_root = data_root
        self.data_split = data_split
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
        if download:
            self.download()
        self.images, self.targets = torch.load(
            os.path.join(self.processed_path, data_file))
        self.classes = [str(k) for k in range(10)]

    def download(self):
        resources = [
            ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
             "f68b3c2dcbeaaa9fbdd348bbdeb94873"],
            ["http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
             "d53e105ee54ea40749a09fcbcd1e9432"],
            ["http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
             "9fb629c4189551a2d022fa330f9573f3"],
            ["http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
             "ec29112dd5afa0611ce80d1b7f02629c"]
        ]
        train_path = os.path.join(self.processed_path, self.train_file)
        test_path = os.path.join(self.processed_path, self.test_file)
        if not os.path.exists(train_path) or not os.path.exists(test_path):
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
                                os.path.join(self.raw_path,
                                             'train-labels-idx1-ubyte')))
            test_set = (self.read_image_file(
                os.path.join(self.raw_path, 't10k-images-idx3-ubyte')),
                        self.read_label_file(
                            os.path.join(self.raw_path,
                                         't10k-labels-idx1-ubyte')))
            with open(os.path.join(self.processed_path,
                                   self.train_file), 'wb') as f:
                torch.save(training_set, f)
            with open(os.path.join(self.processed_path,
                                   self.test_file), 'wb') as f:
                torch.save(test_set, f)
            print('Download complete')

    def read_label_file(self, path):
        with open(path, 'rb') as f:
            x = self.read_sn3_pascalvincent_tensor(f, strict=False)
        assert x.dtype == torch.uint8
        assert x.ndimension() == 1
        return x.long()

    def read_image_file(self, path):
        with open(path, 'rb') as f:
            x = self.read_sn3_pascalvincent_tensor(f, strict=False)
        assert x.dtype == torch.uint8
        assert x.ndimension() == 3
        return x

    def read_sn3_pascalvincent_tensor(self,
                                      path,
                                      strict=True):
        torch_type_map = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')
        }
        with self.open_compressed_file(path) as f:
            data = f.read()
        magic = int(codecs.encode(data[0:4], 'hex'), 16)
        nd = magic % 256
        ty = magic // 256
        assert 1 <= nd <= 3
        assert 8 <= ty <= 14
        m = torch_type_map[ty]
        s = []
        for k in range(nd):
            s.append(int(codecs.encode(
                data[4 * (k + 1): 4 * (k + 2)], 'hex'), 16))
        parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
        assert parsed.shape[0] == np.prod(s) or not strict
        return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

    @staticmethod
    def open_compressed_file(path):
        if not isinstance(path, torch._six.string_classes):
            return path
        if path.endswith('.gz'):
            import gzip
            return gzip.open(path, 'rb')
        if path.endswith('.xz'):
            import lzma
            return lzma.open(path, 'rb')
        return open(path, 'rb')

    def data_length(self):
        return len(self.images)

    def get_image(self, index):
        # Convert to [H, W, C] format
        return self.images[index].numpy()[:, :, None]

    def get_target(self, index):
        return int(self.targets[index])


class SVHN(_BaseDataset):
    def __init__(self,
                 data_root,
                 data_split,
                 input_size,
                 data_augment=False,
                 hyp_params=None,
                 download=True):
        super(SVHN, self).__init__(input_size=input_size,
                                   data_augment=data_augment,
                                   hyp_params=hyp_params)
        assert data_split in ['train', 'val', 'test']
        resources = {
            'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                      "train_32x32.mat",
                      "e26dedcc434d2e4c54c9b2d4a06d8373"],
            'val': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                    "test_32x32.mat",
                    "eb5a983be6a315427106f1b164d9cef3"],
            'test': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                     "extra_32x32.mat",
                     "a93ce644f1a588dc4d68dda5feec44a7"],
        }
        self.data_root = data_root
        self.data_split = data_split
        file_url = resources[data_split][0]
        file_name = resources[data_split][1]
        file_md5 = resources[data_split][2]
        file_path = os.path.join(data_root, file_name)
        if download:
            if not self.check_integrity(file_path, file_md5):
                self.download_url(file_url,
                                  data_root,
                                  file_name,
                                  file_md5)
                print('Download complete')
        loaded_mat = sio.loadmat(os.path.join(data_root, file_name))
        # Convert to [B, H, W, C] format
        self.images = np.transpose(loaded_mat['X'], (3, 0, 1, 2))
        # Loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat['y'].astype(np.int64).squeeze()
        # The svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.classes = [str(k) for k in range(10)]

    def data_length(self):
        return len(self.images)

    def get_image(self, index):
        return self.images[index]

    def get_target(self, index):
        return int(self.targets[index])


class CIFAR10(_BaseDataset):
    base_path = 'cifar-10-batches-py'
    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = "cifar-10-python.tar.gz"
    file_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    data_meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self,
                 data_root,
                 data_split,
                 input_size,
                 data_augment=False,
                 hyp_params=None,
                 download=True):
        super(CIFAR10, self).__init__(input_size=input_size,
                                      data_augment=data_augment,
                                      hyp_params=hyp_params)
        assert data_split in ['train', 'val', 'test']
        self.data_root = data_root
        self.data_split = data_split
        if download:
            if not self._check_integrity():
                self.download_and_extract_archive(self.data_url,
                                                  data_root,
                                                  filename=self.file_name,
                                                  md5=self.file_md5)
                print('Download complete')
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted')
        if data_split == 'train':
            data_list = self.train_list
        else:
            data_list = self.test_list
        images, targets = [], []
        # Load the picked numpy arrays
        for file_name, checksum in data_list:
            file_path = os.path.join(data_root,
                                     self.base_path,
                                     file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                images.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])
        images = np.vstack(images).reshape((-1, 3, 32, 32))
        # Convert to [H, W, C] format
        self.images = images.transpose((0, 2, 3, 1))
        self.targets = targets
        # Load class names
        file_path = os.path.join(data_root,
                                 self.base_path,
                                 self.data_meta['filename'])
        if not self.check_integrity(file_path, self.data_meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted')
        with open(file_path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.data_meta['key']]

    def _check_integrity(self):
        for fentry in (self.train_list + self.test_list):
            file_name, md5 = fentry[0], fentry[1]
            file_path = os.path.join(self.data_root,
                                     self.base_path,
                                     file_name)
            if not self.check_integrity(file_path, md5):
                return False
        return True

    def data_length(self):
        return len(self.images)

    def get_image(self, index):
        return self.images[index]

    def get_target(self, index):
        return int(self.targets[index])


class CIFAR100(CIFAR10):
    base_path = 'cifar-100-python'
    data_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    file_name = "cifar-100-python.tar.gz"
    file_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    data_meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class ImageFolder(_BaseDataset):
    def __init__(self,
                 data_root,
                 data_split,
                 input_size,
                 data_augment=False,
                 hyp_params=None):
        super(ImageFolder, self).__init__(input_size=input_size,
                                          data_augment=data_augment,
                                          hyp_params=hyp_params)
        assert data_split in ['train', 'val', 'test']
        self.data_root = data_root
        self.data_split = data_split
        data_path = os.path.join(data_root, data_split)
        extensions = ('.jpg', '.jpeg', '.png', '.ppm',
                      '.bmp', '.pgm', '.tif', '.tiff')
        classes, class_map = self.find_classes(data_path)
        samples = self.make_dataset(data_path,
                                    class_map,
                                    extensions)
        if len(samples) == 0:
            raise RuntimeError('Found 0 files in ' + data_path)
        self.extensions = extensions
        self.classes = classes
        self.class_map = class_map
        self.samples = samples
        self.targets = [item[1] for item in samples]

    @staticmethod
    def find_classes(data_root):
        classes = [d.name for d in os.scandir(data_root) if d.is_dir()]
        classes.sort()
        class_map = {cls_name: k for k, cls_name in enumerate(classes)}
        return classes, class_map

    @staticmethod
    def make_dataset(directory,
                     class_map,
                     extensions):
        instances = []
        directory = os.path.expanduser(directory)
        for target_class in sorted(class_map.keys()):
            class_index = class_map[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, file_names in sorted(os.walk(target_dir,
                                                      followlinks=True)):
                for name in sorted(file_names):
                    path = os.path.join(root, name)
                    if path.lower().endswith(extensions):
                        instances.append((path, class_index))
        return instances

    def data_length(self):
        return len(self.samples)

    def get_image(self, index):
        file_path = self.samples[index][0]
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError('Missing image %s' % file_path)
        # Convert BGR to RGB format
        image = image[:, :, ::-1]
        return image

    def get_target(self, index):
        return int(self.targets[index])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../datasets',
                        help='data root')
    parser.add_argument('--data_type', type=str, default='mnist',
                        help='data type')
    parser.add_argument('--data_split', type=str, default='train',
                        help='data split')
    parser.add_argument('--input_size', type=int, default=28,
                        help='input size')
    opt = parser.parse_args()

    dataloader = DataLoader(opt.data_root,
                            opt.data_type,
                            opt.data_split,
                            opt.input_size,
                            batch_size=16,
                            download=True)
    print('Batch of dataloader', len(dataloader))
    images, targets = next(iter(dataloader))
    print(images.shape, targets.shape)
