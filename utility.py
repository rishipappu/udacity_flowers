import torch
import numpy as np
import pandas as pd
from torch import nn, optim
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms, models
from PIL import Image


def data_loader(data_dir):
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    # Resize, transform, and normalize images into PyTorch tensors for training, testing, and validation

    image_transforms = [
        transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    ]

    image_sets = [
        datasets.ImageFolder(train_dir, transform=image_transforms[0]),
        datasets.ImageFolder(test_dir, transform=image_transforms[1]),
        datasets.ImageFolder(valid_dir, transform=image_transforms[1]),
    ]

    images = []
    for image_set in image_sets:
        images.append(
            torch.utils.data.DataLoader(image_set, batch_size=64, shuffle=True)
        )

    return images


# Training function written in Udacity's Introduction to Neural Networks with PyTorch (Inference and Validation Notebook) and modified for train.py
def train_model(epochs, model, train_set, device, optimizer, criterion):
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in train_set:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                print(f"Training loss: {running_loss/len(train_set)}")

            model.train()


# Test function written in Udacity's Introduction to Neural Networks with PyTorch (Inference and Validation Notebook) and modified for train.py
def test_model(epochs, model, validation_set, test_set, device, optimizer, criterion):
    print("Evaluating model...\n")
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in validation_set:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_set:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Validation loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(test_set):.3f}.. "
                    f"Test accuracy: {accuracy/len(test_set)*100:.3f}%"
                )
                running_loss = 0
                model.eval()


def save_checkpoint(model, arch, inputs, outputs, hidden_units, save):
    checkpoint = {
        "input_size": inputs,
        "output_size": outputs,
        "state_dict": model.state_dict(),
        "hidden_units": hidden_units,
        "model": arch,
    }
    torch.save(checkpoint, save + "/checkpoint.pth")
    print("Checkpoint saved at " + save + "checkpoint.pth")


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """

    with Image.open(image) as im:
        (width, height) = im.size
        modifier = min(width, height) / 256
        (rwidth, rheight) = (int(width / modifier), int(height / modifier))
        im = im.resize((rwidth, rheight))
        left = (rwidth - 224) // 2
        upper = (rheight - 224) // 2
        right = left + 224
        lower = upper + 224
        im = im.crop((left, upper, right, lower))
        np_image = np.array(im) / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        return np_image.transpose((2, 0, 1))


def predict(image_path, model, device, topk=5):
    """Predict the class (or classes) of an image using a trained deep learning model."""

    image = torch.from_numpy(np.float32(process_image(image_path))).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logps = model(image.unsqueeze(0))
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        return (top_p.detach().cpu().numpy(), top_class.detach().cpu().numpy())
