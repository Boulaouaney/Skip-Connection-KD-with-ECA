import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import models.resnet_ECA_parallel_SC as resnet_ECA_parallel
import models.resnet_last_down_extract as resnet_down_origin
import models.resnet_ECA_last_block_SC as resnet_ECA_last
import models.resnet as resnet

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from progress_bar import progress_bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model on a tensorboard")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="---Model type: resnet18, resnet34, resnet50---",
    )
    parser.add_argument(
        "--pair_keys",
        type=int,
        required=True,
        help="---Indicate pair of keys unique for teacher and student---",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="student",
        help="---Choose the model either teacher or student---",
    )
    parser.add_argument(
        "--ECA", type=str, default="yes", help="---model with ECA or without"
    )
    parser.add_argument(
        "--ECA_block",
        type=str,
        default="parallel",
        help="---ECA block in parallel or last",
    )
    args, unparsed = parser.parse_known_args()

    model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    device = "cuda:0"

    def build_model_ECA_parallel():
        return resnet_ECA_parallel.__dict__[args.model]()

    def build_model_ECA_last():
        return resnet_ECA_last.__dict__[args.model]()

    def build_model_origin():
        return resnet_down_origin.__dict__[args.model]()

    array = torch.randn((32, 256, 8, 8))
    transform_test = transforms.Compose(
        [
            # transforms.Grayscale(),
            transforms.ToTensor()
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10", train=False, download=True, transform=transform_test
    )

    testLoader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    writer = SummaryWriter(
        f"runs/accuracy_check/{args.model}_{args.type}_{args.pair_keys}"
    )

    criterion = nn.CrossEntropyLoss()

    def matplotlib_imshow(img):
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def images_to_probs(net, images):
        """
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        """
        output = net(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output[1], 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        return preds, [
            F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output[1])
        ]

    def plot_classes_preds(net, images, labels):
        """
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        """
        preds, probs = images_to_probs(net, images)
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(12, 12))
        for idx in np.arange(5):
            ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])
            matplotlib_imshow(images[idx])
            ax.set_title(
                "{0}, {1:.1f}%\n(label: {2})".format(
                    classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]]
                ),
                color=("green" if preds[idx] == labels[idx].item() else "red"),
            )
        return fig

    if args.ECA == "no":
        net = build_model_origin()
    elif args.ECA == "yes":
        if args.ECA_block == "parallel":
            # net = resnet.resnet18()
            net = build_model_ECA_parallel()
        elif args.ECA_block == "last":
            net = build_model_ECA_last()

    dataiter = iter(testLoader)
    images, labels = next(dataiter)
    writer.add_graph(net, images)

    net.to(device)
    net.eval()
    net.load_state_dict(
        torch.load(
            f"./saved_pth_model/{args.model}_{args.type}_{args.pair_keys}.pth",
            map_location=torch.device("cuda:0"),
        )
    )
    print(net)
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(testLoader):
            data, target = data.to(device), target.to(device)

            output_1, output = net(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            output = net(data)
            _, predicted = output[1].max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            writer.add_figure(
                "predictions vs. actuals", plot_classes_preds(net, data, target)
            )
            progress_bar(
                batch_idx,
                len(testLoader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (val_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

    writer.close()
