import re
import os
import random
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        # load images from '/path/to/data/trainA'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size),
                              key=lambda f: int(re.findall('\d+', f)[-1]))
        # load images from '/path/to/data/trainB'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size),
                              key=lambda f: int(re.findall('\d+', f)[-1]))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), fixed=True) # do not random crop or resize trainB

        # read in paths for precomputed optical flows and confidence masks
        self.dir_flows = os.path.join(opt.dataroot, 'flows')  # create a path '/path/to/data/flows'
        flow_paths = [os.path.join(self.dir_flows, f)
                      for f in os.listdir(self.dir_flows)
                      if os.path.isfile(os.path.join(self.dir_flows, f)) and f.endswith('pt')]
        flow_paths.sort(key=lambda f: int(re.findall('\d+', f)[-1]))
        self.flow_paths = flow_paths
        self.flow_size = len(self.flow_paths)

        self.dir_confidences = os.path.join(opt.dataroot, 'confidences')
        confidence_paths = [os.path.join(self.dir_confidences, f)
                            for f in os.listdir(self.dir_confidences)
                            if os.path.isfile(os.path.join(self.dir_confidences, f)) and f.endswith('pt')]
        confidence_paths.sort(key=lambda f: int(re.findall('\d+', f)[-1]))
        self.confidence_paths = confidence_paths
        self.confidence_size = len(self.confidence_paths)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths, B_paths, flows and confs
            A (tensor)            -- an image in the input domain
            B (tensor)            -- its corresponding image in the target domain
            A_paths (str)         -- image paths
            B_paths (str)         -- image paths
            flows (tensor)        -- precomputed dense optical flow from frame index to index+1, shape=(H, W, 2)
            confidences (tensor)  -- confidence mask of optical flow, shape=(1, H, W)
        """
        # We want B to be serial, but don't want fixed pairs, so shuffle A
        index_A = random.randint(0, self.A_size - 1)
        A_path = self.A_paths[index_A]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        flow_path = self.flow_paths[index % self.flow_size]
        flow = torch.load(flow_path)
        confidence_path = self.confidence_paths[index % self.confidence_size]
        confidence = torch.load(confidence_path)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'flows': flow, 'confidences': confidence}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
