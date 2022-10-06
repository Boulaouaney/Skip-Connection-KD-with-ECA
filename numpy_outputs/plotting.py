import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Plot teacher and student models accuracy and loss graph')
parser.add_argument('--plot_type', type=str, required=True, help='---Plot type: accuracy or loss')
parser.add_argument('--base', type=str, required=True)
# parser.add_argument('--pair_keys', type=int, required=True, help='---Indicate pair of keys unique for teacher and student---')
parser.add_argument('--pair_keys', type=int, help='---Indicate pair of keys unique for teacher and student---')
args, unparsed = parser.parse_known_args()

if args.base == 'no':
    accuracy_np = np.load(f'./train_accuracy_teacher.npy')
    accuracy_np_val = np.load(f'./val_accuracy_teacher.npy')
    accuracy_np_kd = np.load(f'./train_accuracy_student.npy')
    accuracy_np_kd_val = np.load(f'./val_accuracy_student.npy')
    accuracy_np = accuracy_np/50000.0
    accuracy_np_val = accuracy_np_val/10000.0
    accuracy_np_kd = accuracy_np_kd/50000.0
    accuracy_np_kd_val = accuracy_np_kd_val/10000.0

    loss_np = np.load(f'./train_loss_teacher.npy')
    loss_np_val = np.load(f'./val_loss_teacher.npy')
    loss_np_kd = np.load(f'./train_loss_student.npy')
    loss_np_kd_val = np.load(f'./val_loss_student.npy')

elif args.base == 'yes':
    accuracy_np = np.load('./train_accuracy_resnet18.npy')
    accuracy_np_val = np.load('./val_accuracy_resnet18.npy')

    loss_np = np.load('./train_loss_resnet18.npy')
    loss_np_val = np.load('./val_loss_resnet18.npy')



#print(accuracy_np)
#print(accuracy_np_val)
if args.plot_type == 'accuracy':
    plt.figure(figsize=(10, 5))
    plt.title("Teacher: Training and Validation Accuracy")
    plt.plot(accuracy_np, 'g', label='Training Accuracy')
    plt.plot(accuracy_np_val, 'b', label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Student: Training and Validation Accuracy")
    plt.plot(accuracy_np_kd, 'g', label='Training Accuracy')
    plt.plot(accuracy_np_kd_val, 'b', label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
elif args.plot_type == 'loss':
    plt.figure(figsize=(10, 5))
    plt.title("Teacher: Training and Validation Loss")
    plt.plot(loss_np, 'g', label='Training Loss')
    plt.plot(loss_np_val, 'b', label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Student: Training and Validation Loss")
    plt.plot(loss_np_kd, 'g', label='Training Loss')
    plt.plot(loss_np_kd_val, 'b', label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()