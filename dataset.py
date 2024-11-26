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

        """
        self.color_name_to_id: {'Beige': 0, 'Black': 1, 'Blue': 2, 'Bronze': 3, 'Brown': 4, 'Burgundy': 5, 'Charcoal': 6, 'Coffee Brown': 7, 'Copper': 8, 'Cream': 9, 'Fluorescent Green': 10, 'Gold': 11, 'Green': 12, 'Grey': 13, 'Grey Melange': 14, 'Khaki': 15, 'Lavender': 16, 'Lime Green': 17, 'Magenta': 18, 'Maroon': 19, 'Mauve': 20, 'Metallic': 21, 'Multi': 22, 'Mushroom Brown': 23, 'Mustard': 24, 'NA': 25, 'Navy Blue': 26, 'Nude': 27, 'Off White': 28, 'Olive': 29, 'Orange': 30, 'Peach': 31, 'Pink': 32, 'Purple': 33, 'Red': 34, 'Rose': 35, 'Rust': 36, 'Sea Green': 37, 'Silver': 38, 'Skin': 39, 'Steel': 40, 'Tan': 41, 'Taupe': 42, 'Teal': 43, 'Turquoise Blue': 44, 'White': 45, 'Yellow': 46}
        self.gender_name_to_id: {'Boys': 0, 'Girls': 1, 'Men': 2, 'Unisex': 3, 'Women': 4}
        self.article_name_to_id: {'Accessory Gift Set': 0, 'Baby Dolls': 1, 'Backpacks': 2, 'Bangle': 3, 'Basketballs': 4, 'Bath Robe': 5, 'Beauty Accessory': 6, 'Belts': 7, 'Blazers': 8, 'Body Lotion': 9, 'Body Wash and Scrub': 10, 'Booties': 11, 'Boxers': 12, 'Bra': 13, 'Bracelet': 14, 'Briefs': 15, 'Camisoles': 16, 'Capris': 17, 'Caps': 18, 'Casual Shoes': 19, 'Churidar': 20, 'Clothing Set': 21, 'Clutches': 22, 'Compact': 23, 'Concealer': 24, 'Cufflinks': 25, 'Cushion Covers': 26, 'Deodorant': 27, 'Dresses': 28, 'Duffel Bag': 29, 'Dupatta': 30, 'Earrings': 31, 'Eye Cream': 32, 'Eyeshadow': 33, 'Face Moisturisers': 34, 'Face Scrub and Exfoliator': 35, 'Face Serum and Gel': 36, 'Face Wash and Cleanser': 37, 'Flats': 38, 'Flip Flops': 39, 'Footballs': 40, 'Formal Shoes': 41, 'Foundation and Primer': 42, 'Fragrance Gift Set': 43, 'Free Gifts': 44, 'Gloves': 45, 'Hair Accessory': 46, 'Hair Colour': 47, 'Handbags': 48, 'Hat': 49, 'Headband': 50, 'Heels': 51, 'Highlighter and Blush': 52, 'Innerwear Vests': 53, 'Ipad': 54, 'Jackets': 55, 'Jeans': 56, 'Jeggings': 57, 'Jewellery Set': 58, 'Jumpsuit': 59, 'Kajal and Eyeliner': 60, 'Key chain': 61, 'Kurta Sets': 62, 'Kurtas': 63, 'Kurtis': 64, 'Laptop Bag': 65, 'Leggings': 66, 'Lehenga Choli': 67, 'Lip Care': 68, 'Lip Gloss': 69, 'Lip Liner': 70, 'Lip Plumper': 71, 'Lipstick': 72, 'Lounge Pants': 73, 'Lounge Shorts': 74, 'Lounge Tshirts': 75, 'Makeup Remover': 76, 'Mascara': 77, 'Mask and Peel': 78, 'Mens Grooming Kit': 79, 'Messenger Bag': 80, 'Mobile Pouch': 81, 'Mufflers': 82, 'Nail Essentials': 83, 'Nail Polish': 84, 'Necklace and Chains': 85, 'Nehru Jackets': 86, 'Night suits': 87, 'Nightdress': 88, 'Patiala': 89, 'Pendant': 90, 'Perfume and Body Mist': 91, 'Rain Jacket': 92, 'Rain Trousers': 93, 'Ring': 94, 'Robe': 95, 'Rompers': 96, 'Rucksacks': 97, 'Salwar': 98, 'Salwar and Dupatta': 99, 'Sandals': 100, 'Sarees': 101, 'Scarves': 102, 'Shapewear': 103, 'Shirts': 104, 'Shoe Accessories': 105, 'Shoe Laces': 106, 'Shorts': 107, 'Shrug': 108, 'Skirts': 109, 'Socks': 110, 'Sports Sandals': 111, 'Sports Shoes': 112, 'Stockings': 113, 'Stoles': 114, 'Suits': 115, 'Sunglasses': 116, 'Sunscreen': 117, 'Suspenders': 118, 'Sweaters': 119, 'Sweatshirts': 120, 'Swimwear': 121, 'Tablet Sleeve': 122, 'Ties': 123, 'Ties and Cufflinks': 124, 'Tights': 125, 'Toner': 126, 'Tops': 127, 'Track Pants': 128, 'Tracksuits': 129, 'Travel Accessory': 130, 'Trolley Bag': 131, 'Trousers': 132, 'Trunk': 133, 'Tshirts': 134, 'Tunics': 135, 'Umbrellas': 136, 'Waist Pouch': 137, 'Waistcoat': 138, 'Wallets': 139, 'Watches': 140, 'Water Bottle': 141, 'Wristbands': 142}
        """
        # print(self.color_name_to_id)
        # print(self.gender_name_to_id)
        # print(self.article_name_to_id)


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
