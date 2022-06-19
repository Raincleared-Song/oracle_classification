from torchvision.transforms import transforms
from data_aug.gaussian_blur import Dilation2d, Erosion2d
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data import oracle_dataset
from data.oracle_transform import naive_transform_fn, change_transform_fn


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        erosion = Erosion2d()
        dilation = Dilation2d()
        data_transforms = transforms.Compose([transforms.Pad(10, fill=255),
                                              transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
                                              # transforms.RandomHorizontalFlip(),
                                              # transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomApply([erosion], p=0.3),
                                              transforms.RandomApply([dilation], p=0.3),
                                              # transforms.RandomGrayscale(p=0.2),
                                              # GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, image_res, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True, download=True,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(32), n_views)
                                                ),

            'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled', download=True,
                                            transform=ContrastiveLearningViewGenerator(
                                                self.get_simclr_pipeline_transform(96), n_views)),

            'oracle': lambda: oracle_dataset.OracleDataset(
                data_path=self.root_folder,
                transform=ContrastiveLearningViewGenerator(change_transform_fn(image_res), n_views)),

            'oracle-test': lambda: oracle_dataset.OracleDataset(
                data_path=self.root_folder, transform=naive_transform_fn(image_res))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
