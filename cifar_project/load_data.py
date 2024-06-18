from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob

label_name = ["airplane", "automobile", "bird",
              "cat", "deer", "dog", "frog",
              "horse", "ship", "truck"]
label_dict = {}
for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert("RGB")


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MyDataSet(Dataset):
    def __init__(self, im_list, transform=None, loader=default_loader):
        super(MyDataSet, self).__init__()
        imgs = []

        for im_item in im_list:
            im_label_name = im_item.split("/")[-1].split("\\")[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob("./cifar-10-batches-py/train/*/*.png")
im_test_list = glob.glob("./cifar-10-batches-py/test/*/*.png")

train_dataset = MyDataSet(im_train_list, transform=train_transform)
test_dataset = MyDataSet(im_test_list, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=10, persistent_workers=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=10, persistent_workers=True)
