import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100

train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


def input_dataset(dataset, noise_type, noise_ratio, transform=True, noise_file=None):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=True,
                                transform=train_cifar10_transform if transform else test_cifar10_transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio,
                                noise_file=noise_file,
                                )
        test_dataset = CIFAR10(root='./data/',
                               download=True,
                               train=False,
                               transform=test_cifar10_transform,
                               noise_type=noise_type,
                               noise_rate=noise_ratio
                               )
        num_classes = 10
        num_training_samples = 50000
        num_testing_samples = 10000

    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root='./data/',
                                 download=True,
                                 train=True,
                                 transform=train_cifar100_transform if transform else test_cifar100_transform,
                                 noise_type=noise_type,
                                 noise_rate=noise_ratio,
                                 noise_file=noise_file
                                 )
        test_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=False,
                                transform=test_cifar100_transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                                )
        num_classes = 100
        num_training_samples = 50000
        num_testing_samples = 10000

    return train_dataset, test_dataset, num_classes, num_training_samples, num_testing_samples


def input_dataset_clip(dataset, noise_type, noise_ratio, transform=None, noise_file=None):
    print(dataset)
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root='~/data/',
                                download=True,
                                train=True,
                                transform=transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio,
                                noise_file=noise_file,
                                )
        test_dataset = CIFAR10(root='~/data/',
                               download=True,
                               train=False,
                               transform=transform,
                               noise_type=noise_type,
                               noise_rate=noise_ratio
                               )
        num_classes = 10
        num_training_samples = len(train_dataset.train_labels)
        num_testing_samples = 10000

    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root='~/data/',
                                 download=True,
                                 train=True,
                                 transform=transform,
                                 noise_type=noise_type,
                                 noise_rate=noise_ratio,
                                 noise_file=noise_file
                                 )
        test_dataset = CIFAR100(root='~/data/',
                                download=True,
                                train=False,
                                transform=transform,
                                noise_type=noise_type,
                                noise_rate=noise_ratio
                                )
        num_classes = 100
        num_training_samples = len(train_dataset.train_noisy_labels)
        num_testing_samples = 10000

    return train_dataset, test_dataset, num_classes, num_training_samples, num_testing_samples
