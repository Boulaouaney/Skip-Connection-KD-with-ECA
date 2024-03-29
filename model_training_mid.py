import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from progress_bar import progress_bar
import models.resnet_ECA_parallel_SC as resnet_ECA_parallel
import models.resnet_last_down_extract as resnet_down_origin
import models.resnet_ECA_last_block_SC as resnet_ECA_last
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys

torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Cifar10 Training with KD and ECA."
    )
    parser.add_argument("--batch", default=32, type=int, help="batch size")
    parser.add_argument(
        "--shuffle", default=True, type=bool, help="shuffle the training dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="---Model type: resnet18, resnet34, resnet50---",
    )
    parser.add_argument(
        "--model_kd",
        type=str,
        required=True,
        help="---Model type: resnet18, resnet34, resnet50---",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--pair_keys",
        type=int,
        required=True,
        help="---Indicate pair of keys unique for teacher and student---",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="---Distillation weight (alpha) (default: 0.3)---",
    )
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--epoch", default=300, type=int, help="epoch number")
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    best_acc = 0
    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []
    train_accuracy_kd = []
    val_accuracy_kd = []
    train_loss_kd = []
    val_loss_kd = []

    def build_model_ECA_parallel():
        return (
            resnet_ECA_parallel.__dict__[args.model](),
            resnet_ECA_parallel.__dict__[args.model_kd](),
        )

    def build_model_ECA_last():
        return (
            resnet_ECA_last.__dict__[args.model](),
            resnet_ECA_last.__dict__[args.model_kd](),
        )

    def build_model_origin():
        return (
            resnet_down_origin.__dict__[args.model](),
            resnet_down_origin.__dict__[args.model_kd](),
        )

    print("Teacher model type: ", args.model)
    print("Student model type: ", args.model_kd)

    print("==> Preparing data..")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10", train=True, download=True, transform=transform_train
    )

    trainLoader = DataLoader(
        trainset, batch_size=args.batch, shuffle=args.shuffle, num_workers=2
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
        f"runs/experiments/{args.model}_{args.model_kd}_{args.ECA}_{args.ECA_block}_{args.pair_keys}"
    )

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

    print("==> Building model..")

    def train(model, loader, optimizer, epoch):
        model.train()

        train_loss = 0
        correct = 0
        total = 0

        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output1, output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # log to tensorboard every 100 mini batches
            running_loss += loss.item()
            if batch_idx % 100 == 99:
                writer.add_scalar(
                    "Teacher/training Loss",
                    running_loss / 100,
                    epoch * len(loader) + batch_idx,
                )

                writer.add_scalar(
                    "Teacher/training accuracy",
                    100.0 * correct / total,
                    epoch * len(loader) + batch_idx,
                )

                writer.add_figure(
                    "Teacher/predictions vs. actuals",
                    plot_classes_preds(model, data, target),
                    global_step=epoch * len(loader) + batch_idx,
                )
                running_loss = 0.0

            progress_bar(
                batch_idx,
                len(trainLoader),
                "Teacher: Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
        return correct, train_loss

    def train_distil(model, distil_model, loader, optimizer, distil_weights, epoch):
        model.eval()
        distil_model.train()

        train_loss_kd = 0
        correct_kd = 0
        total_kd = 0

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output1_t, output_t = model(data)
            output1_s, output_s = distil_model(data)

            kd_loss = F.mse_loss(output1_s, output1_t.detach()) * distil_weights
            kd_loss_cls = criterion(output_s, target)
            loss_kd = kd_loss + kd_loss_cls
            loss_kd.backward()
            optimizer.step()

            train_loss_kd += loss_kd.item()
            _, predicted = output_s.max(1)
            total_kd += target.size(0)
            correct_kd += predicted.eq(target).sum().item()

            # log to tensorboard every 100 mini batches
            running_loss += loss_kd.item()
            if batch_idx % 100 == 99:
                writer.add_scalar(
                    "Student/training Loss",
                    running_loss / 100,
                    epoch * len(loader) + batch_idx,
                )
                writer.add_scalar(
                    "Student/training accuracy",
                    100.0 * correct_kd / total_kd,
                    epoch * len(loader) + batch_idx,
                )

                writer.add_figure(
                    "Student/predictions vs. actuals",
                    plot_classes_preds(distil_model, data, target),
                    global_step=epoch * len(loader) + batch_idx,
                )
                running_loss = 0.0

            progress_bar(
                batch_idx,
                len(trainLoader),
                "Student: Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss_kd / (batch_idx + 1),
                    100.0 * correct_kd / total_kd,
                    correct_kd,
                    total_kd,
                ),
            )
        return correct_kd, train_loss_kd

    def validate(model, loader, who, epoch):
        model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)

                output_1, output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                writer.add_scalar(
                    f"{who}/validation Loss", val_loss / (batch_idx + 1), epoch
                )
                writer.add_scalar(
                    f"{who}/validation accuracy", 100.0 * correct / total, epoch
                )

                progress_bar(
                    batch_idx,
                    len(testLoader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        val_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

        return correct, val_loss

    if args.ECA == "no":
        models_teacher, distil_models = build_model_origin()
    elif args.ECA == "yes":
        if args.ECA_block == "parallel":
            models_teacher, distil_models = build_model_ECA_parallel()
        elif args.ECA_block == "last":
            models_teacher, distil_models = build_model_ECA_last()

    models_teacher = models_teacher.to(device)
    distil_models = distil_models.to(device)

    dataiter = iter(trainLoader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    writer.add_graph(models_teacher, images)
    writer.add_graph(distil_models, images)

    criterion = nn.CrossEntropyLoss()
    optimizer_teacher = optim.SGD(
        models_teacher.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_teacher, T_max=200
    )

    optimizer_student = optim.SGD(
        distil_models.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_student, T_max=200
    )

    distil_weight = args.alpha
    best_loss = sys.maxsize
    best_loss_kd = sys.maxsize

    print("Training Teacher first... =====>")
    for epoch in range(args.epoch):
        print("\nTeacher Epoch: %d" % epoch)
        train_correct, training_loss = train(
            models_teacher, trainLoader, optimizer_teacher, epoch
        )
        val_correct, validating_loss = validate(
            models_teacher, testLoader, "Teacher", epoch
        )
        scheduler_teacher.step()
        train_accuracy.append(train_correct)
        train_loss.append(training_loss)

        val_accuracy.append(val_correct)
        val_loss.append(validating_loss)
        torch.cuda.empty_cache()
        if epoch >= 0 and (validating_loss - best_loss) < 0:
            best_loss = validating_loss
            torch.save(
                models_teacher.state_dict(),
                f"./saved_pth_model/{args.model}_teacher_{args.pair_keys}.pth",
            )
    print("\n")

    print("Saving Teacher Numpy Outputs... =====>")
    train_accuracy_np = np.asarray(train_accuracy)
    train_loss_np = np.asarray(train_loss)

    val_accuracy_np = np.asarray(val_accuracy)
    val_loss_np = np.asarray(val_loss)

    np.save(
        f"./numpy_outputs/train_accuracy_teacher_{args.pair_keys}", train_accuracy_np
    )
    np.save(f"./numpy_outputs/train_loss_teacher_{args.pair_keys}", train_loss_np)

    np.save(f"./numpy_outputs/val_accuracy_teacher_{args.pair_keys}", val_accuracy_np)
    np.save(f"./numpy_outputs/val_loss_teacher_{args.pair_keys}", val_loss_np)

    print("Training Student second... =====>")
    for epoch in range(args.epoch):
        print("\nStudent Epoch: %d" % epoch)
        train_correct_kd, training_loss_kd = train_distil(
            models_teacher,
            distil_models,
            trainLoader,
            optimizer_student,
            distil_weight,
            epoch,
        )
        val_correct_kd, validating_loss_kd = validate(
            distil_models, testLoader, "Student", epoch
        )
        scheduler_student.step()

        train_accuracy_kd.append(train_correct_kd)
        train_loss_kd.append(training_loss_kd)

        val_accuracy_kd.append(val_correct_kd)
        val_loss_kd.append(validating_loss_kd)
        torch.cuda.empty_cache()
        if epoch >= 0 and (validating_loss_kd - best_loss_kd) < 0:
            best_loss_kd = validating_loss_kd
            torch.save(
                distil_models.state_dict(),
                f"./saved_pth_model/{args.model_kd}_student_{args.pair_keys}.pth",
            )

    writer.close()

    print("Saving Student Numpy Outputs... =====>")
    train_accuracy_np_kd = np.asarray(train_accuracy_kd)
    train_loss_np_kd = np.asarray(train_loss_kd)

    val_accuracy_np_kd = np.asarray(val_accuracy_kd)
    val_loss_np_kd = np.asarray(val_loss_kd)

    np.save(
        f"./numpy_outputs/train_accuracy_student_{args.pair_keys}", train_accuracy_np_kd
    )
    np.save(f"./numpy_outputs/train_loss_student_{args.pair_keys}", train_loss_np_kd)

    np.save(
        f"./numpy_outputs/val_accuracy_student_{args.pair_keys}", val_accuracy_np_kd
    )
    np.save(f"./numpy_outputs/val_loss_student_{args.pair_keys}", val_loss_np_kd)
