import os.path
import time

import torch
import torch.utils.data as data
from PIL import Image, ImageEnhance, ImageOps
import random
from random import randrange
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def select_k_items(lst, k):
    if k > len(lst):
        raise ValueError("k cannot be greater than the length of the list.")
    if k == len(lst):
        return lst
    # random.seed(time.time())
    indices = random.sample(range(len(lst)), k)
    return [lst[i] for i in indices]


class TrainLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/low')
        self.dir_B = os.path.join(self.root, self.phase + '/high')


        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

    def __getitem__(self, index):
        input_name = self.A_paths[index]
        ps = self.fineSize
        inp_img = Image.open(self.A_paths[index]).convert("RGB")
        tar_img = Image.open(self.B_paths[index]).convert("RGB")

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        hh, ww = inp_img.shape[1], inp_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]



        return inp_img, tar_img, input_name



    def __len__(self):
        return len(self.A_paths)


class TrainUnlabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize, k):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/low')



        # image path
        self.A_paths = select_k_items(sorted(make_dataset(self.dir_A)), k)



    def __getitem__(self, index):

        input_name = self.A_paths[index]
        ps = self.fineSize
        inp_img = Image.open(self.A_paths[index]).convert("RGB")
        # inp_img = data_aug(inp_img)

        w, h = inp_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')


        inp_img = TF.to_tensor(inp_img)

        hh, ww = inp_img.shape[1], inp_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)


        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]

        return inp_img, input_name

    def __len__(self):
        return len(self.A_paths)


class ValLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize
        self.mul = 8
        self.dir_A = os.path.join(self.root, self.phase + '/low')
        self.dir_B = os.path.join(self.root, self.phase + '/high')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))



    def __getitem__(self, index):

        inp_img = Image.open(self.A_paths[index]).convert("RGB")
        tar_img = Image.open(self.B_paths[index]).convert("RGB")

        w, h = inp_img.size

        H, W = ((h + self.mul) // self.mul) * self.mul, ((w + self.mul) // self.mul) * self.mul
        padh = H - h if h % self.mul != 0 else 0
        padw = W - w if w % self.mul != 0 else 0
        inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)


        return inp_img, tar_img, h, w, self.A_paths[index].split('/')[-1], self.B_paths[index]



    def __len__(self):
        return len(self.A_paths)


class TestData(data.Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.root = dataroot
        self.mul = 8


        self.dir_A = os.path.join(self.root)

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))

    def __getitem__(self, index):
        inp_img = Image.open(self.A_paths[index]).convert("RGB")

        w, h = inp_img.size

        H, W = ((h + self.mul) // self.mul) * self.mul, ((w + self.mul) // self.mul) * self.mul
        padh = H - h if h % self.mul != 0 else 0
        padw = W - w if w % self.mul != 0 else 0
        inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
        inp_img = TF.to_tensor(inp_img)



        return inp_img,  h, w


    def __len__(self):
        return len(self.A_paths)



