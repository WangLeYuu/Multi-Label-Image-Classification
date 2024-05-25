import argparse


def get_args():
    parser = argparse.ArgumentParser(description='ALL ARGS')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=100, help='Total Training Times')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of processes to handle dataset loading')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--datasets_path', type=str, default='./datasets/', help='Path to the dataset')
    parser.add_argument('--image_path', type=str, default='./datasets/images', help='Path to the style image')
    parser.add_argument('--original_csv_path', type=str, default='./datasets/styles.csv', help='Original csv file dir')
    parser.add_argument('--train_csv_path', type=str, default='./datasets/train.csv', help='train csv file dir')
    parser.add_argument('--val_csv_path', type=str, default='./datasets/val.csv', help='val csv file dir')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='log dir')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='checkpoints dir')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/2024-05-24_13-50/checkpoint-000002.pth', help='choose a checkpoint to predict')
    parser.add_argument('--predict_image_path', type=str, default='./datasets/images/1163.jpg', help='show ground truth')

    return parser.parse_args()
