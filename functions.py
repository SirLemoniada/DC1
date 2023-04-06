from pathlib import Path
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
import torch
from net import Net
from typing import Callable, List, Generator, Tuple
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random
from image_dataset import ImageDataset

class BatchSampler:
    """
    Implements an iterable which given a torch dataset and a batch_size
    will produce batches of data of that given size. The batches are
    returned as tuples in the form (images, labels).
    Can produce balanced batches, where each batch will have an equal
    amount of samples from each class in the dataset. If your dataset is heavily
    imbalanced, this might mean throwing away a lot of samples from
    over-represented classes!
    """

    def __init__(self, batch_size: int, dataset: ImageDataset, balanced: bool = False) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.balanced = balanced
        if self.balanced:
            # Counting the ocurrence of the class labels:
            unique, counts = np.unique(self.dataset.targets, return_counts=True)
            indexes = []
            # Sampling an equal amount from each class:
            for i in range(len(unique)):
                indexes.append(
                    np.random.choice(
                        np.where(self.dataset.targets == i)[0],
                        size=counts.min(),
                        replace=False,
                    )
                )
            # Setting the indexes we will sample from later:
            self.indexes = np.concatenate(indexes)
        else:
            # Setting the indexes we will sample from later (all indexes):
            self.indexes = [i for i in range(len(dataset))]

    def __len__(self) -> int:
        return (len(self.indexes) // self.batch_size) + 1

    def shuffle(self) -> None:
        random.shuffle(self.indexes)

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        remaining = False
        self.shuffle()
        # Go over the datset in steps of 'self.batch_size':
        for i in range(0, len(self.indexes), self.batch_size):
            # If our current batch is larger than the remaining data, we quit:
            if i + self.batch_size > len(self.indexes):
                remaining = True
                break
            # If not, we yield a complete batch:
            else:
                # Getting a list of samples from the dataset, given the indexes we defined:
                X_batch = [
                    self.dataset[self.indexes[k]][0]
                    for k in range(i, i + self.batch_size)
                ]
                Y_batch = [
                    self.dataset[self.indexes[k]][1]
                    for k in range(i, i + self.batch_size)
                ]
                # Stacking all the samples and returning the target labels as a tensor:
                yield torch.stack(X_batch).float(), torch.tensor(Y_batch).long()
        # If there is still data left that was not a full batch:
        if remaining:
            # Return the last batch (smaller than batch_size):
            X_batch = [
                self.dataset[self.indexes[k]][0] for k in range(i, len(self.indexes))
            ]
            Y_batch = [
                self.dataset[self.indexes[k]][1] for k in range(i, len(self.indexes))
            ]
            yield torch.stack(X_batch).float(), torch.tensor(Y_batch).long()


def rotate_images(path: Path):
  images_array=np.load(path)
  rotated_images=[]
  for img in images_array:
    for degrees in range(0,10):
      rotated_image=rotate(img[0], 36*degrees, reshape=False)
      rotated_images.append(rotated_image)

  # pic=Image.fromarray(rotated_images[0])
  # pic.show()
  return rotated_images

def flip_images(path: Path):
  images_array=np.load(path)
  flipped_images=[]
  for img in images_array:
    pic=np.fliplr(img[0])
    flipped_images.append(pic)

  # pic=Image.fromarray(flipped_images[0])
  # pic.show()
  return flipped_images

def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Lets keep track of all the losses:
    losses = []
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)

        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()
    return losses


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []

    # We will save the actual and predicted values for analytics
    actual_values = np.array([])
    prediction_values = np.array([])

    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)

            actual_values = np.append(actual_values, y.cpu().detach().numpy())
            prediction_values = np.append(prediction_values, prediction.argmax(dim=1).cpu().detach().numpy())

            losses.append(loss)

    return losses, actual_values, prediction_values

def test_model_original(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []
    predictions = []
    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)
            predicted_class = torch.argmax(prediction, dim=1)
            predictions.append(predicted_class)
            predictions.append(y)
    return losses, predictions
