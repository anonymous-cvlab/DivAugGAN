"""
prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os
import torchvision.models as models
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import sklearn.metrics
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob


# __all__ = ['compute_prdc']

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        # self.fnames = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.fnames = glob(os.path.join(root, '**', '*.jpg'), recursive=True) + \
            glob(os.path.join(root, '**', '*.png'), recursive=True)

        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)


class FileNames(Dataset):
    def __init__(self, fnames, transform=None):
        self.fnames = fnames
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)


def get_custom_loader(image_dir_or_fnames, image_size=224, batch_size=50, num_workers=4, num_samples=-1):    
    transform = []
    transform.append(transforms.Resize([image_size, image_size]))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    transform = transforms.Compose(transform)
    if isinstance(image_dir_or_fnames, list): 
        dataset = FileNames(image_dir_or_fnames, transform=transform)
    elif isinstance(image_dir_or_fnames, str): 
        dataset = ImageFolder(image_dir_or_fnames, transform=transform)
    else:
        raise TypeError
    
    if num_samples > 0:
        dataset.fnames = dataset.fnames[:num_samples] 
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return data_loader


class PRDC():
    def __init__(self, batch_size=50, k=3, num_samples=10000, model=None):
        self.manifold_ref = None
        self.batch_size = batch_size
        self.k = k
        self.num_samples = num_samples
        if model is None:
            print('loading vgg16 for improved precision/recall and density/coverage ...', end='', flush=True)
            self.vgg16 = models.vgg16(pretrained=True).cuda().eval()
            print('done')
        else: 
            self.vgg16 = model   
            
    # def __call__(self, subject):
    #     return self.precision_and_recall(subject)
    
    def extract_features_from_files(self, path_or_fnames): 
        """
        Extract features of vgg16-fc2 for all images in path
        params:
            path_or_fnames: dir containing images or list of fnames(str)
        returns:
            A numpy array of dimension (num images, dims)
        """
        dataloader = get_custom_loader(path_or_fnames, batch_size=self.batch_size, num_samples=self.num_samples)
        num_found_images = len(dataloader.dataset)
        desc = 'extracting features of {:d} images'.format(num_found_images)          
        if num_found_images < self.num_samples: 
            print('WARNING: num_found_images {:d} < num_samples {:d}'.format(num_found_images, self.num_samples))
        features = [] 
        for batch in tqdm(dataloader, desc=desc): 
            before_fc = self.vgg16.features(batch.cuda())  
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:4](before_fc)
            features.append(feature.cpu().data.numpy())
        return np.concatenate(features, axis=0)    
    
    # def get_kth_value(self, unsorted, k, axis=-1):
    #     """
    #     Args:
    #         unsorted: numpy.ndarray of any dimensionality.
    #         k: int
    #     Returns:
    #         kth values along the designated axis.
    #     """
    #     indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    #     k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    #     kth_values = k_smallests.max(axis=axis)
    #     return kth_values
    
    def compute_prdc(self, real_features, fake_features, nearest_k):
        """
        Computes precision, recall, density, and coverage given two manifolds.

        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """

        print('Num real: {} Num fake: {}'.format(real_features.shape[0], fake_features.shape[0]))
        real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
        fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake_features, nearest_k)
        distance_real_fake = compute_pairwise_distance(real_features, fake_features)
        precision = (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()
        recall = (distance_real_fake <np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()
        density = (1.0 / float(nearest_k)) * (distance_real_fake <np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0).mean()
        coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()
        return dict(precision=precision, recall=recall, density=density, coverage=coverage)
        

        
        

        
        


 


# def compute_prdc(real_features, fake_features, nearest_k):
#     """
#     Computes precision, recall, density, and coverage given two manifolds.

#     Args:
#         real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
#         fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
#         nearest_k: int.
#     Returns:
#         dict of precision, recall, density, and coverage.
#     """

#     print('Num real: {} Num fake: {}'
#           .format(real_features.shape[0], fake_features.shape[0]))

#     real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
#         real_features, nearest_k)
#     fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
#         fake_features, nearest_k)
#     distance_real_fake = compute_pairwise_distance(
#         real_features, fake_features)

#     precision = (
#             distance_real_fake <
#             np.expand_dims(real_nearest_neighbour_distances, axis=1)
#     ).any(axis=0).mean()

#     recall = (
#             distance_real_fake <
#             np.expand_dims(fake_nearest_neighbour_distances, axis=0)
#     ).any(axis=1).mean()

#     density = (1. / float(nearest_k)) * (
#             distance_real_fake <
#             np.expand_dims(real_nearest_neighbour_distances, axis=1)
#     ).sum(axis=0).mean()

#     coverage = (
#             distance_real_fake.min(axis=1) <
#             real_nearest_neighbour_distances
#     ).mean()

#     return dict(precision=precision, recall=recall,
#                 density=density, coverage=coverage)
