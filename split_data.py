import csv
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from option import get_args


def save_csv(data, path, fieldnames=['image_path', 'gender', 'articleType', 'baseColour']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))


if __name__ == '__main__':

    args = get_args()
    input_folder = args.datasets_path
    output_folder = args.datasets_path

    annotation = args.original_csv_path

    all_data = []
    with open(annotation) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in tqdm(reader, total=reader.line_num):
            img_id = row['id']

            # only three attributes are used: gender articleType、baseColour
            gender = row['gender']
            articleType = row['articleType']
            baseColour = row['baseColour']
            img_name = os.path.join(input_folder, 'images', str(img_id) + '.jpg')

            # Determine if the image exists
            if os.path.exists(img_name):

                # Check if the image is 80 * 60 size and if it is in RGB format
                img = Image.open(img_name)
                if img.size == (60, 80) and img.mode == "RGB":
                    all_data.append([img_name, gender, articleType, baseColour])

    np.random.seed(42)

    all_data = np.asarray(all_data)

    # Randomly select 40000 data points
    inds = np.random.choice(40000, 40000, replace=False)

    # Divide training and validation sets
    save_csv(all_data[inds][:32000], args.train_csv_path)
    save_csv(all_data[inds][32000:40000], args.val_csv_path)
