import argparse
import torch
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
import json
from utility import *

parser = argparse.ArgumentParser()

parser.add_argument("input")
parser.add_argument("checkpoint")
parser.add_argument("--top_k", help="Number of result predictions", default=5, type=int)
parser.add_argument(
    "--category_names", help="A JSON document mapping categories to indices"
)
parser.add_argument(
    "--gpu",
    help="Choose to train the model on the GPU.",
    action="store_true" if torch.cuda.is_available() else "store_false",
)

args = parser.parse_args()

device = torch.device("cuda:0" if args.gpu else "cpu")

# Load the model from the checkpoint

checkpoint = torch.load(args.checkpoint)
model = torch.hub.load("pytorch/vision", checkpoint["model"])
model.classifier = nn.Sequential(
    nn.Linear(25088, checkpoint["hidden_units"]),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(checkpoint["hidden_units"], 102),
    nn.LogSoftmax(dim=1),
)
model.load_state_dict(checkpoint["state_dict"])


# Attach a json fle correlating labels to the image data

with open(args.category_names, "r") as f:
    cat_to_name = json.load(f)

probs, classes = predict(args.input, model, device, args.top_k)

flowers = []
for flower in classes.flatten():
    flowers.append(cat_to_name[str(flower)])

print(probs.flatten())
print(flowers)
