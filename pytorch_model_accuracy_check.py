import torchvision
import torchvision.transforms as transforms
import argparse
from progress_bar import progress_bar
import models.resnet_check as resnet
import models.resnet_ECA_parallel_SC as resnet_ECA_parallel
import models.resnet_last_down_extract as resnet_down_origin
import models.resnet_ECA_last_block_SC as resnet_ECA_last
import models.resnet_teacher as teacher
import models.resnet_student as student
import models.resnet_student_all_fm as student_fm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchsummary import summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checking the PyTorch model accuracy')
    parser.add_argument('--model', type=str, required=True, help='---Model type: resnet18, resnet34, resnet50---')
    parser.add_argument('--pair_keys', type=int, required=True,
                        help='---Indicate pair of keys unique for teacher and student---')
    parser.add_argument('--type', type=str, default='student', help='---Choose the model either teacher or student---')
    parser.add_argument('--ECA', type=str, default='yes', help='---model with ECA or without')
    parser.add_argument('--ECA_block', type=str, default='parallel', help='---ECA block in parallel or last')
    args, unparsed = parser.parse_known_args()

    model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    #device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    device = 'cuda:0'

    def build_model_ECA_parallel():
        if args.model == 'resnet18':
            return resnet_ECA_parallel.__dict__[model_names[0]]()
        elif args.model == 'resnet34':
            return resnet_ECA_parallel.__dict__[model_names[1]]()
        elif args.model == 'resnet50':
            return resnet_ECA_parallel.__dict__[model_names[2]]()
        elif args.model == 'resnet101':
            return resnet_ECA_parallel.__dict__[model_names[3]]()
        elif args.model == 'resnet152':
            return resnet_ECA_parallel.__dict__[model_names[4]]()

    def build_model_ECA_last():
        if args.model == 'resnet18':
            return resnet_ECA_last.__dict__[model_names[0]]()
        elif args.model == 'resnet34':
            return resnet_ECA_last.__dict__[model_names[1]]()
        elif args.model == 'resnet50':
            return resnet_ECA_last.__dict__[model_names[2]]()
        elif args.model == 'resnet101':
            return resnet_ECA_last.__dict__[model_names[3]]()
        elif args.model == 'resnet152':
            return resnet_ECA_last.__dict__[model_names[4]]()
    def build_model_origin():
        if args.model == 'resnet18':
            return resnet_down_origin.__dict__[model_names[0]]()
        elif args.model == 'resnet34':
            return resnet_down_origin.__dict__[model_names[1]]()
        elif args.model == 'resnet50':
            return resnet_down_origin.__dict__[model_names[2]]()
        elif args.model == 'resnet101':
            return resnet_down_origin.__dict__[model_names[3]]()
        elif args.model == 'resnet152':
            return resnet_down_origin.__dict__[model_names[4]]()



    array = torch.randn((32, 256, 8, 8))
    transform_test = transforms.Compose([
        #transforms.Grayscale(),
        transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR10(
        root='./cifar10', train=False, download=True, transform=transform_test)

    testLoader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    #number_of_classes = 10
    #confusion_matrix = torch.zeros(number_of_classes, number_of_classes)


    if args.ECA == 'no':
        net = build_model_origin().to(device)
        net.eval()
        summary(net, (3, 32, 32))
        net.load_state_dict(torch.load(f'./vanilla_kd_model_saved_base/{args.model}_{args.type}_{args.pair_keys}.pth',
                                       map_location=torch.device('cuda:0')))

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

                progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    elif args.ECA == 'yes':
        if args.ECA_block == 'parallel':
            net = build_model_ECA_parallel().to(device)

            summary(net, (3, 32, 32))
            net.load_state_dict(torch.load(f'./vanilla_kd_model_saved_base/{args.model}_{args.type}_{args.pair_keys}.pth',
                                           map_location=torch.device('cuda:0')))
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

                    progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        elif args.ECA_block == 'last':
            net = build_model_ECA_last().to(device)

            summary(net, (3, 32, 32))
            net.load_state_dict(torch.load(f'./vanilla_kd_model_saved_base/{args.model}_{args.type}_{args.pair_keys}.pth',
                                           map_location=torch.device('cuda:0')))

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

                    progress_bar(batch_idx, len(testLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))
