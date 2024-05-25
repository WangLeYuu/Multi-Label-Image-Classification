import torch
import torch.nn as nn
import torchvision.models as models


class MultiOutputModel(nn.Module):
    def __init__(self, n_color_classes, n_gender_classes, n_article_classes):
        super().__init__()

        self.base_model = models.mobilenet_v2().features
        last_channel = models.mobilenet_v2().last_channel

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Create three independent classifiers for predicting three categories
        self.color = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=n_color_classes))
        self.gender = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=n_gender_classes))
        self.article = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=n_article_classes))

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        return {
            'color': self.color(x),
            'gender': self.gender(x),
            'article': self.article(x)
        }
