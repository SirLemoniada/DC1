# Custom imports
from image_dataset import ImageDataset
from net import Net
from functions import *

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

#Minor preprocessing
#import all of the data sets.
X_test = np.load('data/X_test.npy')
X_train = np.load('data/X_train.npy')
y_test = np.load('data/y_test.npy')
y_train = np.load('data/y_train.npy')

#Since we use kfold we dont need holdouts. Thus we merge datasets.
combined_x = np.concatenate((X_test, X_train))
combined_y = np.concatenate((y_test, y_train))
#We reset the y labels

def main(statistics_df, actual_prediction_df, fold_iteration, args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test data set
    train_dataset = ImageDataset(Path('data/training_set_x.npy'), Path('data/training_set_y.npy'))
    test_dataset = ImageDataset(Path('data/test_set_x.npy'), Path('data/test_set_y.npy'))

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.01)
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
#    elif (
 #       torch.backends.mps.is_available() and not DEBUG
  #  ):  # PyTorch supports Apple Silicon GPU's from version 1.12
   #     print("@@@ Apple silicon device enabled, training with Metal backend...")
    #    device = "mps"
     #   model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=128, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=128, dataset=test_dataset, balanced=args.balanced_batches
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []

    for e in range(n_epochs):
        if activeloop:

            # Training:
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            training_mean_loss = mean_loss.cpu().detach().numpy()

            # Testing:
            losses, actual_values, prediction_values = test_model(model, test_sampler, loss_function, device)

            # # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            testing_mean_loss = mean_loss.cpu().detach().numpy()



            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])


    confusion_matrix_output = confusion_matrix(actual_values, prediction_values)
    confusion_matrix_df = pd.DataFrame(confusion_matrix_output)


    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

    # Create plot of losses
    fig = figure(figsize=(9, 4), dpi=80)

    plt.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    plt.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()

    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")
    return(statistics_df, confusion_matrix_df, actual_prediction_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=10, type=int)
    parser.add_argument("--batch_size", help="batch_size", default=128, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=False,
        type=bool,
    )
    parser.add_argument("--training_set_x", help="Training images", default=np.load('data/X_train.npy'), type=int)
    parser.add_argument("--training_set_y", help="Training labels", default=np.load('data/y_train.npy'), type=int)
    parser.add_argument("--test_set_x", help="Testing images", default=np.load('data/X_test.npy'), type=int)
    parser.add_argument("--test_set_y", help="Testing labels", default=np.load('data/y_test.npy'), type=int)

    args = parser.parse_args()

    #We are going to create a pandas df for analytics
    statistics_df = pd.DataFrame()
    actual_prediction_df = pd.DataFrame()


    #Here we define a validation state, such that if true it runs k fold validation, else a normal train on preset train/test split.
    validation_state = False
    if validation_state == True:

        #Here we begin defining the k_fold

        kf = KFold(n_splits=10)

        #assign our dataset to the fold function
        for i, (train_index, test_index) in enumerate(kf.split(combined_x)):
            print(i, train_index, test_index)
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

            #we now create an independant numpy array for each fold
            training_set_x = combined_x[train_index]
            np.save('data/training_set_x', training_set_x)

            training_set_y = combined_y[train_index]
            np.save('data/training_set_y', training_set_y)

            test_set_x = combined_x[test_index]
            np.save('data/test_set_x', test_set_x)

            test_set_y = combined_y[test_index]
            np.save('data/test_set_y', test_set_y)

            #start running
            statistics_df, confusion_matrix_df, actual_prediction_df = main(statistics_df, actual_prediction_df, i, args)

    else:
        i=1
        #we set the training sets to be the presets
        training_set_x = np.load('data/X_train.npy')
        np.save('data/training_set_x', training_set_x)

        training_set_y = np.load('data/y_train.npy')
        np.save('data/training_set_y', training_set_y)

        test_set_x = np.load('data/X_test.npy')
        np.save('data/test_set_x', test_set_x)

        test_set_y = np.load('data/y_test.npy')
        np.save('data/test_set_y', test_set_y)

        #start running
        statistics_df, confusion_matrix_df, actual_prediction_df = main(statistics_df, actual_prediction_df, i, args)


# statistics_df.to_csv('data/stats.csv')
confusion_matrix_df.to_csv('data/confusion_matrix.csv')
