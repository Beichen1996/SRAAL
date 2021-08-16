from torchvision import datasets, transforms


def cifar10_transformer_train():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914009, 0.48215896, 0.44653079], std=[0.24703279, 0.24348423, 0.26158753]),
])
def cifar10_transformer_test():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914009, 0.48215896, 0.44653079], std=[0.24703279, 0.24348423, 0.26158753]),
])

def cifar100_transformer_train():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50707543, 0.48655024, 0.44091907], std=[0.26733398, 0.25643876, 0.27615029]),
])
def cifar100_transformer_test():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50707543, 0.48655024, 0.44091907], std=[0.26733398, 0.25643876, 0.27615029]),
])


def imagenet_transformer():
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
