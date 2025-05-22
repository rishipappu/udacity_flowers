# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

### train.py

`train.py` contains a generic script to download and train any pre-trained model in PyTorch's library on a dataset of flower images with customizable parameters.
|Flags| Input|Default Value|
|-----|------|-------|
|`datadir`|The directory containing images split into `train`, `test`, and `valid` directories for their respective ImageLoaders| `N/A`|
|`--learning-rate`|Set the learning rate for the classifier|`0.003`|
|`--arch`|Choose the pre-trained model on which to train your classifier| `vgg16`|
|`--hidden-units`|Set the number of hidden units within the classifier| `16580`|
|`--save-dir`| Set the directory to save the classifier checkpoint| `./checkpoints`|
|`--gpu`| If you have have a CUDA compatible gpu this flag will use the GPU for training| If GPU is available it will be the default|
|`--epochs`| The number of epochs to train the model| `10`|

### predict.py

`predict.py` contains a generic script to determine the most probable label for an image provided a pre-trained model, an image, and JSON file of label mappings.
|Flags| Input|Default Value|
|-----|------|-------|
|`--input`|The input image to classify|`N/A`|
|`--checkpoint`|A file containing a checkpoint for a previously trained model|`N/A`|
|`--top-k`|Determines the top `k` number of probable results for the provided image|`5`|
|`--category_names`|A JSON file that contains a number of label mappings for images| `N/A`|
|`--gpu`| If you have have a CUDA compatible gpu this flag will use the GPU for training| If GPU is available it will be the default|

### Image Classifier Project

`Image Classifier Project.ipynb` contains a Jupyter Notebook containing a project that implements both `train.py` and `predict.py` with static values to identify and display the top 5 values for a given image with a 70% accuracy.
