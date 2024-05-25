import os
from datetime import datetime
import warnings
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn.functional as F


# Get the current date and time and format it as a string
def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model, f)
    print('Saved checkpoint:', f)


# Load Checkpoints
def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model = torch.load(name, map_location='cpu')
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return model, epoch


def get_loss(net_output, ground_truth):
    color_loss = F.cross_entropy(net_output['color'], ground_truth['color_labels'])
    gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender_labels'])
    article_loss = F.cross_entropy(net_output['article'], ground_truth['article_labels'])

    loss = color_loss + gender_loss + article_loss

    return loss, {'color': color_loss, 'gender': gender_loss, 'article': article_loss}


def calculate_metrics(output, target):
    _, predicted_color = output['color'].cpu().max(1)
    gt_color = target['color_labels'].cpu()

    _, predicted_gender = output['gender'].cpu().max(1)
    gt_gender = target['gender_labels'].cpu()

    _, predicted_article = output['article'].cpu().max(1)
    gt_article = target['article_labels'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_color = balanced_accuracy_score(y_true=gt_color.numpy(), y_pred=predicted_color.numpy())
        accuracy_gender = balanced_accuracy_score(y_true=gt_gender.numpy(), y_pred=predicted_gender.numpy())
        accuracy_article = balanced_accuracy_score(y_true=gt_article.numpy(), y_pred=predicted_article.numpy())

    return accuracy_color, accuracy_gender, accuracy_article
