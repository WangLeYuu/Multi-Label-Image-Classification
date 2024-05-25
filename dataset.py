import csv
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
from option import get_args

args = get_args()

mean = [0.85418772, 0.83673165, 0.83065592]
std = [0.25331535, 0.26539705, 0.26877365]


class AttributesDataset():
    def __init__(self, annotation_path):
        color_labels = []
        gender_labels = []
        article_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                color_labels.append(row['baseColour'])
                gender_labels.append(row['gender'])
                article_labels.append(row['articleType'])

        # Remove duplicate values to obtain a unique label set
        self.color_labels = np.unique(color_labels)
        self.gender_labels = np.unique(gender_labels)
        self.article_labels = np.unique(article_labels)

        # Calculate the number of categories for each label
        self.num_colors = len(self.color_labels)
        self.num_genders = len(self.gender_labels)
        self.num_articles = len(self.article_labels)

        # Create label mapping: Create two dictionaries: one from label ID to label name, and the other from label name to label ID.
        # Mapping results：self.gender_name_to_id：{'Boys': 0, 'Girls': 1, 'Men': 2, 'Unisex': 3, 'Women': 4}
        # Mapping results.gender_id_to_name：{0: 'Boys', 1: 'Girls', 2: 'Men', 3: 'Unisex', 4: 'Women'}
        self.color_id_to_name = dict(zip(range(len(self.color_labels)), self.color_labels))
        self.color_name_to_id = dict(zip(self.color_labels, range(len(self.color_labels))))

        self.gender_id_to_name = dict(zip(range(len(self.gender_labels)), self.gender_labels))
        self.gender_name_to_id = dict(zip(self.gender_labels, range(len(self.gender_labels))))

        self.article_id_to_name = dict(zip(range(len(self.article_labels)), self.article_labels))
        self.article_name_to_id = dict(zip(self.article_labels, range(len(self.article_labels))))


class FashionDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # Initialize a list to store the image path and corresponding labels of the dataset
        self.data = []
        self.color_labels = []
        self.gender_labels = []
        self.article_labels = []

        # Read data from a CSV file and store the image path and corresponding labels in a list
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.color_labels.append(self.attr.color_name_to_id[row['baseColour']])
                self.gender_labels.append(self.attr.gender_name_to_id[row['gender']])
                self.article_labels.append(self.attr.article_name_to_id[row['articleType']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        dict_data = {
            'img': img,
            'labels': {
                'color_labels': self.color_labels[idx],
                'gender_labels': self.gender_labels[idx],
                'article_labels': self.article_labels[idx]
            }
        }
        return dict_data


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# Calculate the mean and variance of all images in the dataset
def get_mean_and_std(image_paths, transform):
    # Initialize the accumulator of mean and variance
    means = np.zeros((3,))
    stds = np.zeros((3,))
    count = 0

    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        image_array = image_tensor.numpy()

        # Calculate the mean and variance of the image
        batch_mean = np.mean(image_array, axis=(0, 2, 3))
        batch_var = np.var(image_array, axis=(0, 2, 3))

        # Accumulate to the total
        means += batch_mean
        stds += batch_var
        count += 1

        # Calculate the mean and standard deviation of the entire dataset
    means /= count
    stds = np.sqrt(stds / count)

    return means, stds


# Calculate the mean and variance of the dataset
if __name__ == '__main__':

    mena_std_transform = transforms.Compose([transforms.ToTensor()])

    image_path = []
    for root, _, files in os.walk(args.image_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'):
                image_path.append(os.path.join(root, file))

    means, stds = get_mean_and_std(image_path, mena_std_transform)
    print("Calculated mean and standard deviation:=========>")
    print("Mean:", means)
    print("Std:", stds)
