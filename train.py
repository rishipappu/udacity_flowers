import argparse
import torch
from torch import nn, optim
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms, models
from utility import *

# Collect all script arguments

parser = argparse.ArgumentParser()

parser.add_argument("data_dir")
parser.add_argument(
    "--learning_rate",
    help="Assign a learning rate for the model.",
    default=0.003,
    type=float,
)
parser.add_argument("--arch", help="Choose a model architecture.", default="vgg16")
parser.add_argument(
    "--hidden_units",
    help="Set the number of hidden_units in the model's classifier.",
    default=16580,
    type=int,
)
parser.add_argument(
    "--save_dir",
    help="Choose a directory to save the model checkpoint.",
    default="./checkpoints",
)
parser.add_argument(
    "--gpu",
    help="Choose to train the model on the GPU.",
    action="store_true" if torch.cuda.is_available() else "store_false",
)
parser.add_argument(
    "--epochs",
    help="Choose the number of epochs to train the model on.",
    default=10,
    type=int,
)


args = parser.parse_args()

# Sort arguments by flag

hidden_units = args.hidden_units
device = torch.device("cuda:0" if args.gpu else "cpu")
model = torch.hub.load("pytorch/vision", args.arch, pretrained=True)

images = data_loader(args.data_dir)

# Freeze parameters for the model so we don't backprop through them

for param in model.parameters():
    param.requires_grad = False


# Build a classifier with the specified number of hidden units

model.classifier = model.classifier = nn.Sequential(
    nn.Linear(25088, hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_units, 102),
    nn.LogSoftmax(dim=1),
)

# Train classifier parameters exclusively

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
criterion = nn.NLLLoss()

if args.gpu:
    print("GPU found, using CUDA...\n")
else:
    print("GPU not found, using CPU...\n")

model.to(device)

# Begin training the model for the specified number of epochs, printing the training loss after every 5 iterations
train_model(args.epochs, model, images[0], device, optimizer, criterion)

# Evaluate the model on validation and test sets
test_model(args.epochs, model, images[2], images[1], device, optimizer, criterion)

# Save the model as a checkpoint
save_checkpoint(model, args.arch, 20588, 102, args.hidden_units, args.save_dir)
