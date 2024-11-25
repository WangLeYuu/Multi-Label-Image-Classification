# Start Tensorboard：tensorboard --logdir=logs/2024-05-24_15-16
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import AttributesDataset, FashionDataset, train_transform, val_transform
from model import MultiOutputModel
from utils import get_loss, get_cur_time, checkpoint_save
from predict import calculate_metrics, validate
from option import get_args

args = get_args()

# Initial parameters
start_epoch = args.start_epoch
N_epochs = args.epochs
batch_size = args.batch_size
num_workers = args.num_workers
batch_size = args.batch_size
device = args.device

# Initial paths
original_csv_path = args.original_csv_path
train_csv_path = args.train_csv_path
val_csv_path = args.val_csv_path
log_dir = args.log_dir
checkpoint_dir = args.checkpoint_dir

# Load attribute classes, The attributes contain labels and mappings for three categories
attributes = AttributesDataset(original_csv_path)

# Load Dataset
train_dataset = FashionDataset(train_csv_path, attributes, train_transform)
val_dataset = FashionDataset(val_csv_path, attributes, val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Load model
model = MultiOutputModel(n_color_classes=attributes.num_colors,
                            n_gender_classes=attributes.num_genders,
                            n_article_classes=attributes.num_articles)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)    # Add learning rate decay

logdir = os.path.join(log_dir, get_cur_time())
savedir = os.path.join(checkpoint_dir, get_cur_time())

os.makedirs(logdir, exist_ok=True)
os.makedirs(savedir, exist_ok=True)


logger = SummaryWriter(logdir)

n_train_samples = len(train_dataloader)


if __name__ == '__main__':

    for epoch in range(start_epoch, N_epochs):

        # Initialize training loss and accuracy for each category
        total_loss, color_loss, gender_loss, article_loss = 0, 0, 0, 0
        accuracy_color, accuracy_gender, accuracy_article = 0, 0, 0

        # Create a tqdm instance to visualize training progress
        pbar = tqdm(total=len(train_dataset), desc='Training', unit='img')

        for batch in train_dataloader:
            pbar.update(train_dataloader.batch_size)    # Update progress bar

            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            # Calculate losses
            loss_train, losses_train = get_loss(output, target_labels)
            total_loss += loss_train.item()
            color_loss += losses_train['color']
            gender_loss += losses_train['gender']
            article_loss += losses_train['article']

            # Calculation accuracy
            batch_accuracy_color, batch_accuracy_gender, batch_accuracy_article = calculate_metrics(output, target_labels)

            accuracy_color += batch_accuracy_color
            accuracy_gender += batch_accuracy_gender
            accuracy_article += batch_accuracy_article

            loss_train.backward()
            optimizer.step()
        sch.step()

        # Print epoch, total loss, loss for each category, accuracy for each category
        print("epoch {:2d}, total_loss: {:.4f}, color_loss: {:.4f}, gender_loss: {:.4f}, article_loss: {:.4f}, color_acc: {:.4f}, gender_acc: {:.4f}, article_acc: {:.4f}".format(
            epoch,
            total_loss / n_train_samples, color_loss / n_train_samples, gender_loss / n_train_samples, article_loss / n_train_samples,
            accuracy_color / n_train_samples, accuracy_gender / n_train_samples, accuracy_article / n_train_samples))

        # Loss and accuracy write to logs
        logger.add_scalar('train_total_loss', total_loss / n_train_samples, epoch)
        logger.add_scalar('train_color_loss', color_loss / n_train_samples, epoch)
        logger.add_scalar('train_gender_loss', gender_loss / n_train_samples, epoch)
        logger.add_scalar('train_article_loss', article_loss / n_train_samples, epoch)
        logger.add_scalar('train_color_acc', accuracy_color / n_train_samples, epoch)
        logger.add_scalar('train_gender_acc', accuracy_gender / n_train_samples, epoch)
        logger.add_scalar('train_article_acc', accuracy_article / n_train_samples, epoch)

        if epoch % 2 == 0:
            validate(model=model, dataloader=val_dataloader, logger=logger, iteration=epoch, device=device, checkpoint=None)

        if epoch % 2 == 0:
            checkpoint_save(model, savedir, epoch)
        pbar.close()
