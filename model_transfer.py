from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import FashionDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from utils import get_loss, calculate_metrics, checkpoint_load
from option import get_args


args = get_args()
batch_size = args.batch_size
num_workers = args.num_workers
device = args.device
original_csv_path = args.original_csv_path
val_csv_path = args.val_csv_path
checkpoint=args.checkpoint
predict_image_path = args.predict_image_path


def validate(model, dataloader, logger, iteration, device, checkpoint):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():

        # The total loss and accuracy of each category in initializing the validation set
        avg_loss, accuracy_color, accuracy_gender, accuracy_article = 0, 0, 0, 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_color, batch_accuracy_gender, batch_accuracy_article = calculate_metrics(output, target_labels)

            accuracy_color += batch_accuracy_color
            accuracy_gender += batch_accuracy_gender
            accuracy_article += batch_accuracy_article

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_color /= n_samples
    accuracy_gender /= n_samples
    accuracy_article /= n_samples
    print('-' * 80)
    print("Validation ====> loss: {:.4f}, color_acc: {:.4f}, gender_acc: {:.4f}, article_acc: {:.4f}\n".format(
        avg_loss, accuracy_color, accuracy_gender, accuracy_article))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_color_acc', accuracy_color, iteration)
    logger.add_scalar('val_gender_acc', accuracy_gender, iteration)
    logger.add_scalar('val_article_acc', accuracy_article, iteration)

    model.train()


def visualize_grid(model, dataloader, attributes, show_cn_matrices=True, show_images=True, checkpoint=None, show_gt=True):
    if checkpoint is not None:
        model, _ = checkpoint_load(model, checkpoint)
    model.eval()

    # Define image list
    imgs = []       

    # Define a list of predicted results (predicted labels, predicted color labels, predicted gender labels, predicted article labels)
    labels, predicted_color_all, predicted_gender_all, predicted_article_all = [], [], [], []

    # Define a list of real values (real labels, real color labels, real gender labels, real article labels)
    gt_labels, gt_color_all, gt_gender_all, gt_article_all = [], [], [], []

    # Initialize precision for each category
    accuracy_color = 0
    accuracy_gender = 0
    accuracy_article = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_colors = batch['labels']['color_labels']
            gt_genders = batch['labels']['gender_labels']
            gt_articles = batch['labels']['article_labels']
            output = model(img)

            batch_accuracy_color, batch_accuracy_gender, batch_accuracy_article = \
                calculate_metrics(output, batch['labels'])
            accuracy_color += batch_accuracy_color
            accuracy_gender += batch_accuracy_gender
            accuracy_article += batch_accuracy_article

            # Calculate maximum probability prediction label
            _, predicted_colors = output['color'].cpu().max(1)
            _, predicted_genders = output['gender'].cpu().max(1)
            _, predicted_articles = output['article'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_color = attributes.color_id_to_name[predicted_colors[i].item()]
                predicted_gender = attributes.gender_id_to_name[predicted_genders[i].item()]
                predicted_article = attributes.article_id_to_name[predicted_articles[i].item()]

                gt_color = attributes.color_id_to_name[gt_colors[i].item()]
                gt_gender = attributes.gender_id_to_name[gt_genders[i].item()]
                gt_article = attributes.article_id_to_name[gt_articles[i].item()]

                gt_color_all.append(gt_color)
                gt_gender_all.append(gt_gender)
                gt_article_all.append(gt_article)

                predicted_color_all.append(predicted_color)
                predicted_gender_all.append(predicted_gender)
                predicted_article_all.append(predicted_article)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_gender, predicted_article, predicted_color))
                gt_labels.append("{}\n{}\n{}".format(gt_gender, gt_article, gt_color))

    if not show_gt:
        n_samples = len(dataloader)
        print("Accuracy ====> color: {:.4f}, gender: {:.4f}, article: {:.4f}".format(
            accuracy_color / n_samples,
            accuracy_gender / n_samples,
            accuracy_article / n_samples))

    # Draw confusion matrix
    if show_cn_matrices:
        # Color confusion matrix
        cn_matrix = confusion_matrix(
            y_true=gt_color_all,
            y_pred=predicted_color_all,
            labels=attributes.color_labels,
            normalize='true')
        ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.color_labels).plot(include_values=False, xticks_rotation='vertical')
        plt.title("Colors")
        plt.tight_layout()
        plt.savefig("confusion_matrix_color.png")
        # plt.show()

        # Gender confusion matrix
        cn_matrix = confusion_matrix(
            y_true=gt_gender_all,
            y_pred=predicted_gender_all,
            labels=attributes.gender_labels,
            normalize='true')
        ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.gender_labels).plot(xticks_rotation='horizontal')
        plt.title("Genders")
        plt.tight_layout()
        plt.savefig("confusion_matrix_gender.png")
        # plt.show()

        # Article confusion matrix (with too many categories, images may be too large to display fully)
        cn_matrix = confusion_matrix(
            y_true=gt_article_all,
            y_pred=predicted_article_all,
            labels=attributes.article_labels,
            normalize='true')
        plt.rcParams.update({'font.size': 1.8})
        plt.rcParams.update({'figure.dpi': 300})
        ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.article_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.rcParams.update({'figure.dpi': 100})
        plt.rcParams.update({'font.size': 5})
        plt.title("Article types")
        plt.savefig("confusion_matrix_article.png")
        # plt.show()

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig("images.png")
        # plt.show()

    model.train()


if __name__ == '__main__':

    """
    Dir testing
    """
    attributes = AttributesDataset(original_csv_path)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = FashionDataset(val_csv_path, attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MultiOutputModel(n_color_classes=attributes.num_colors, n_gender_classes=attributes.num_genders, n_article_classes=attributes.num_articles).to('cpu')

    visualize_grid(model, test_dataloader, attributes, show_cn_matrices=True, show_images=True, checkpoint=checkpoint, show_gt=False)

    """
    Single image testing
    """
    model = torch.load(checkpoint, map_location='cpu')
    img = Image.open(predict_image_path)  
    if img.mode != 'RGB':  
        img = img.convert('RGB')  
    img_tensor = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_color = outputs['color'].cpu().max(1)
        _, predicted_gender = outputs['gender'].cpu().max(1)
        _, predicted_article = outputs['article'].cpu().max(1)

        print("Predicted 【color: {}】, 【gender: {}】, 【article: {}】".format(
            attributes.color_id_to_name[predicted_color.item()],
            attributes.gender_id_to_name[predicted_gender.item()],
            attributes.article_id_to_name[predicted_article.item()]))

