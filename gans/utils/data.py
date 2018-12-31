import os
import torch
from PIL import Image
import csv
from torch.utils import data


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir_, extensions):
    images = []
    attributes = None
    n_variants = None
    dir_ = os.path.expanduser(dir_)
    for sub_dir in os.listdir(dir_):
        d = os.path.join(dir_, sub_dir)
        for root, _, fnames in list(os.walk(d)):
            for imgfile in fnames:
                if has_file_allowed_extension(imgfile, extensions):
                    imgpath = os.path.join(root, imgfile)
                    csvpath = os.path.splitext(imgpath)[0] + '.csv'
                    if os.path.isfile(csvpath):
                        attributes_, variants, n_variants_ = target_loader(csvpath)
                        if attributes is None:
                            attributes, n_variants = attributes_, n_variants_
                        elif attributes != attributes_ or n_variants != n_variants_:
                            raise ValueError('Variants are not consistant at file: {}'.format(csvpath))
                        images.append((imgpath, torch.tensor(variants)))
    return images, attributes, torch.tensor(n_variants)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def target_loader(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        vals = [(key, int(val), int(val_n)) for key, val, val_n in reader]
    return zip(*vals)


class CartoonSet(data.Dataset):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    def __init__(self, root, transform=None, target_transform=None):
        samples, attributes, n_variants = make_dataset(root, self.IMG_EXTENSIONS)
        if len(samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: {}\nSupported extensions are: {}".format(root,
                                                                                                         ', '.join(self.IMG_EXTENSIONS)))

        self.root = root
        self.imgloader = default_loader

        self.samples = samples
        self.attributes = attributes
        self.n_variants = n_variants

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        imgpath, attributes = self.samples[index]
        sample = self.imgloader(imgpath)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            attributes = self.target_transform(attributes)

        return sample, attributes

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
