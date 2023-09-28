import os
import json
from collections import OrderedDict
from torchvision import  transforms
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, GTSRB, Food101, SUN397, EuroSAT, UCF101, StanfordCars, Flowers102, DTD, OxfordIIITPet, MNIST, ImageNet, ImageFolder
import numpy as np
from PIL import Image
import lmdb
import pickle
import six
import os


'''
    function for loading datasets
    contains: 
        1. CIFAR-10
        2. CIFAR-100   
        3. SVHN
        4. GTSRB
        5. FOOD-101
        6. SUN-397
        7. EUROSAT
        8. UCF-101
        9. Stanford Cars
        10. FLOWERS-102
        11. DTD
        12. Oxford Pets
        13. MNIST
        14. ImageNet
'''
IMAGENETNORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}


def get_model(args):
    torch.hub.set_dir('./cache')
    # network
    if args.network == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device)
    elif args.network == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(args.device)
    elif args.network == "instagram":
        from torch import hub
        network = hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(args.device)
    elif args.network == 'vgg':
        from torchvision.models import vgg16, VGG16_Weights, vgg16_bn
        # network = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(args.device)
        network = vgg16_bn(pretrained=True)
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    
    return network


def image_transform(args, transform_type):
    normalize = transforms.Normalize(mean=IMAGENETNORMALIZE['mean'], std=IMAGENETNORMALIZE['std'])
    if transform_type == 'vp':
        train_transform = transforms.Compose([
            transforms.Resize((int(args.input_size*9/8), int(args.input_size*9/8))),
            transforms.RandomCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
        ])
    elif transform_type == 'ff':
        train_transform = transforms.Compose([
            transforms.Resize((252,252)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            normalize
        ])
    return train_transform, test_transform, normalize


def get_torch_dataset(args, transform_type):
    data_path = os.path.join(args.data, args.dataset)
    dataset = args.dataset
    train_transform, test_transform, normalize = image_transform(args, transform_type)
    if args.prune_method=='vpns' or 'vp' in args.prune_mode:
        val_transform = test_transform
    else:
        val_transform = train_transform

    if dataset == "cifar10":
        train_set = CIFAR10(data_path, train=True, transform=train_transform, download=True)
        val_set = CIFAR10(data_path, train=True, transform=val_transform, download=True)
        test_set = CIFAR10(data_path, train=False, transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == "cifar100":
        train_set = CIFAR100(data_path, train=True, transform=train_transform, download=True)
        val_set = CIFAR100(data_path, train=True, transform=val_transform, download=True)
        test_set = CIFAR100(data_path, train=False, transform=test_transform, download=True)
        class_cnt = 100

    elif dataset == "svhn":
        train_set = SVHN(data_path, split = 'train', transform=train_transform, download=True)
        val_set = SVHN(data_path, split = 'train', transform=val_transform, download=True)
        test_set = SVHN(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == "gtsrb":
        full_data = GTSRB(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(GTSRB(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(GTSRB(data_path, split = 'train', transform=val_transform, download=True), val_indices)
        test_set = GTSRB(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 43

    elif dataset == 'food101':
        train_set = Food101(data_path, split = 'train', transform=train_transform, download=True)
        val_set = Food101(data_path, split = 'train', transform=val_transform, download=True)
        test_set = Food101(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 101

    elif dataset == 'sun397':
        img_dir = os.path.join(data_path, 'SUN397')
        train_partition_file = os.path.join(data_path, 'Partitions/Training_01.txt')
        test_partition_file = os.path.join(data_path, 'Partitions/Testing_01.txt')
        target_file = os.path.join(data_path, 'Partitions/ClassName.txt')
        full_data = SUN397Dataset(img_dir = img_dir, partition_file = train_partition_file,  target_file=target_file)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(SUN397Dataset(img_dir = img_dir, partition_file = train_partition_file, target_file=target_file, transform=train_transform), train_indices)
        val_set = Subset(SUN397Dataset(img_dir = img_dir, partition_file = train_partition_file, target_file=target_file, transform=val_transform), val_indices)
        test_set = SUN397Dataset(img_dir = img_dir, partition_file = test_partition_file, target_file=target_file, transform=test_transform)
        class_cnt = 397

    elif dataset == 'eurosat':
        full_data = EuroSAT(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(EuroSAT(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(EuroSAT(data_path, split = 'train', transform=val_transform, download=True), val_indices)
        test_set = EuroSAT(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == 'ucf101':
        annotation_path = os.path.join(data_path, 'ucfTrainTestlist')
        data_path = os.path.join(data_path, 'UCF-101')
        full_data = UCF101(root = data_path,  annotation_path=annotation_path, frames_per_clip=1, fold=1, train = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(UCF101(data_path, train = True, annotation_path=annotation_path, frames_per_clip=1, fold=1, transform=train_transform), train_indices)
        val_set = Subset(UCF101(data_path, train = True, annotation_path=annotation_path, frames_per_clip=1, fold=1, transform=val_transform), val_indices)
        test_set = UCF101(data_path, train = False, annotation_path=annotation_path, frames_per_clip=1, fold=1, transform=test_transform)
        class_cnt = 101
    
    elif dataset == 'stanfordcars':
        data_path = os.path.join(data_path, 'car_data/car_data')
        train_set = ImageFolder(data_path+'/train/', transform=train_transform)
        val_set = ImageFolder(data_path+'/train/', transform=val_transform)
        test_set = ImageFolder(data_path+'/test/', transform=test_transform)
        class_cnt = 196
    
    elif dataset == 'flowers102':
        train_set = ConcatDataset([COOPLMDBDataset(root = data_path, split="train", transform = train_transform), \
                                   COOPLMDBDataset(root = data_path, split="val", transform = train_transform)])
        val_set = ConcatDataset([COOPLMDBDataset(root = data_path, split="val", transform = val_transform), \
                                 COOPLMDBDataset(root = data_path, split="train", transform = val_transform)])
        test_set = COOPLMDBDataset(root = data_path, split="test", transform = test_transform)
        class_cnt = 102
    
    elif dataset == 'dtd':
        train_set = ConcatDataset([DTD(root = data_path, split = 'train', transform=train_transform, download = True), \
                                DTD(root = data_path, split = 'val', transform=train_transform, download = True)])
        val_set = ConcatDataset([DTD(root = data_path, split = 'val', transform=val_transform, download = True), \
                                 DTD(root = data_path, split = 'train', transform=val_transform, download = True)])
        test_set = DTD(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 47

    elif dataset == 'oxfordpets':
        train_set = OxfordIIITPet(data_path, split = 'trainval', transform=train_transform, download=True)
        val_set = OxfordIIITPet(data_path, split = 'trainval', transform=val_transform, download=True)
        test_set = OxfordIIITPet(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 37
    
    elif dataset == 'mnist':
        train_set = MNIST(data_path, train = True, transform=train_transform, download=True)
        val_set = MNIST(data_path, train = True, transform=val_transform, download=True)
        test_set = MNIST(data_path, train = False, transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == 'imagenet':
        imagenet_path = args.imagenet_path
        train_set = ImageFolder(os.path.join(imagenet_path, 'train'), transform=train_transform)
        val_set = ImageFolder(os.path.join(imagenet_path, 'train'), transform=val_transform)
        test_set = ImageFolder(os.path.join(imagenet_path, 'val'), transform=test_transform)
        class_cnt = 1000
    
    elif dataset == 'tiny_imagenet':
        train_set = ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/train'), transform=train_transform)
        val_set = ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/train'), transform=val_transform)
        test_set = TinyImageNet(os.path.join(data_path, 'tiny-imagenet-200/val/images'), os.path.join(data_path, 'tiny-imagenet-200/val/val_annotations.txt'), 
                                os.path.join(data_path, 'tiny-imagenet-200/wnids.txt'), transform=test_transform)
        class_cnt = 200

    else:
        raise NotImplementedError(f"{dataset} not supported")
    
    if dataset == 'imagenet':
        train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True)
    elif dataset in ['dtd', 'oxfordpets']:
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif dataset in ['flowers102', 'stanfordcars']:
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    args.class_cnt = class_cnt
    args.normalize = normalize
    print(f'Dataset information: {dataset}\t {len(train_set)} images for training \t {len(val_set)} images for validation\t')
    print(f'{len(test_set)} images for testing\t')
    return train_loader, val_loader, test_loader


def get_indices(full_data):
    full_len = len(full_data)
    train_len = int(full_len * 0.9)
    indices = np.random.permutation(full_len)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    return train_indices, val_indices


def choose_dataloader(args):
    if args.prune_method=='vpns' or 'vp' in args.prune_mode:
        print('choose visual prompt dataset')
        train_loader, val_loader, test_loader = get_torch_dataset(args, 'vp')       
    else:
        print('choose full finetune dataset')
        train_loader, val_loader, test_loader = get_torch_dataset(args, 'ff')
    return train_loader, val_loader, test_loader


class SubsetWithTransform(Subset):
    def __init__(self, dataset, indices, transform=None):
        super(SubsetWithTransform, self).__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y


class SUN397Dataset(Dataset):
    def __init__(self, img_dir, partition_file, target_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(target_file, 'r') as f:
            self.label_names = [l.strip() for l in f.readlines()]
        self.label_idx = {name: idx for idx,name in enumerate(self.label_names)}

        self.img_names = []
        self.targets = []
        with open(partition_file, 'r') as f:
            lines = f.readlines()
        for l in lines:
            l = l.strip()
            self.img_names.append(l)
            label_name, _ = os.path.split(l)
            self.targets.append(self.label_idx[label_name])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.img_dir+img_name
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = self.targets[idx]
        return image, target


class TinyImageNet(Dataset):
    def __init__(self, root_dir, annotations_file, label_ids_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.entries = open(annotations_file).read().strip().split('\n')

        with open(label_ids_file, 'r') as f:
            self.label_names = [l.strip() for l in f.readlines()]
        self.label_names = sorted(self.label_names)
        self.label_idx = {name: idx for idx,name in enumerate(self.label_names)}

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        line = self.entries[index].split('\t')
        img_path, annotation = line[0], line[1]
        image = Image.open(self.root_dir + '/' + img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        return image, int(self.label_idx[annotation])


class LMDBDataset(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__()
        db_path = os.path.join(root, f"{split}.lmdb")
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = pickle.loads(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf)

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class COOPLMDBDataset(LMDBDataset):
    def __init__(self, root, split="train", transform=None) -> None:
        super().__init__(root, split, transform=transform)
        with open(os.path.join(root, "split.json")) as f:
            split_file = json.load(f)
        idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split_file["test"]}.items()))
        self.classes = list(idx_to_class.values())


class ReverseImageFolder(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        return super().__getitem__(len(self) - 1 - index)  # This reverses the order of items