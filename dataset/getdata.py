"""
CUB200-2011
Get dataset and dataloader in terms of that having train_list.txt and test_list.txt
with the form <image_name class_id> for each line
and class_map.txt with <class_id class_name>
"""
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class MyData(Dataset):
    def __init__(self, root_dir, list_txt, tolength, aug=False, corrupt=None):
        super(MyData, self).__init__()
        self.imglabels = pd.read_table(os.path.join(root_dir, list_txt),
                                       delim_whitespace=True, header=None)
        self.rootdir = os.path.join(root_dir, "images/")    # according to the directory tree
        self.tolength = tolength
        normalizer = tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if aug:
            self.transformer = tv.transforms.Compose([
                tv.transforms.ToPILImage('RGB'),
                tv.transforms.RandomResizedCrop(tolength, scale=(0.4, 1.0)),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                normalizer,
            ])
        else:
            self.transformer = tv.transforms.Compose([
                tv.transforms.ToPILImage('RGB'),
                tv.transforms.Resize(tolength),
                tv.transforms.CenterCrop(tolength),
                # tv.transforms.Resize((tolength, tolength)),
                tv.transforms.ToTensor(),
                normalizer,
            ])
        self.length = self.imglabels.shape[0]
        self.corrupt = corrupt
        if corrupt is not None:
            idx_bin = torch.randint(0, self.length, (int(self.length * corrupt),)).tolist()
            new_labels = torch.randint(0, 200, (int(self.length * corrupt),)).tolist()
            self.false_dict = dict(zip(idx_bin, new_labels))

    def __len__(self):
        return self.length

    def _transform(self, image):
        image = image[:, :, ::-1]  # Convert BGR (cv2) to RGB
        return self.transformer(image)

    def __getitem__(self, item):    # Get through here every epoch when refreshing the dataloader
        img_name, label = self.imglabels.iloc[item, :]
        if self.corrupt is not None and item in self.false_dict.keys():
            label = self.false_dict[item] if self.false_dict[item] != label else (label + 1) % 200
        img_dir = os.path.join(self.rootdir, img_name)
        image = cv2.imread(img_dir, 1)
        image = self._transform(image)
        return image, label


def getloader(root_dir, list_txt, tolength, shuffle=True, bs=16, aug=False, worker=4, corrupt=None):
    mydataset = MyData(root_dir, list_txt, tolength, aug=aug, corrupt=corrupt)
    loader = DataLoader(mydataset, batch_size=bs, shuffle=shuffle, num_workers=worker)
    return loader


if __name__ == "__main__":
    root_dir = "/workspace/datasets/CUB_200_2011/CUB_200_2011"
    train_list, test_list = "train_list.txt", "test_list.txt"
    class_map = pd.read_table(os.path.join(root_dir, 'class_map.txt'), header=None, delim_whitespace=True)
    trainset = True
    tolength = 224  # input image is of size 224*224
    list_txt = train_list if trainset else test_list
    myloader = getloader(root_dir, list_txt, tolength, shuffle=False, bs=16, aug=trainset)
    # display some results
    print(len(myloader))
    dataiter = iter(myloader)
    images, label = dataiter.next()
    print(images.size(), images[2].size(), label, class_map.iloc[label[2].item(), 1])
    images = tv.utils.make_grid(images)
    images = (images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) +
              torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) * 255
    cv2.imwrite("batch.jpg", images.numpy().transpose(1, 2, 0)[:, :, ::-1])
    print("Note: batch display saved in file batch.jpg")
    print(' '.join('%s' % class_map.iloc[label[j].item(), 1] for j in range(16)))
