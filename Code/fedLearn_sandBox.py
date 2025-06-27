###
# main.py
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Federated Learning SandBox
# From: https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html
###

#pip install flwr[simulation] flwr-datasets[vision]
from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, Subset

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation # Run local
from flwr_datasets import FederatedDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
#disable_progress_bar()

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`

##  LOAD the DATASET
NUM_CLIENTS = 3 #Fed learn clients 
                 # Each client gets 1/NUM_CLIENTS % of the data
BATCH_SIZE = 16

# This downloads and caches the dataset locally
from torchvision.datasets import CIFAR10
transform = transforms.ToTensor()
CIFAR10(root="./data", train=True, download=True, transform=transform)
CIFAR10(root="./data", train=False, download=True, transform=transform)

### Depricated
def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    #full_train_dataset = CIFAR10(root="./data", train=True, download=False, transform=transform)``
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader


# No, don't load the full dataset
def load_and_partition_cifar10():
    transform = transforms.ToTensor()
    full_train_dataset = CIFAR10(root="./data", train=True, download=False, transform=transform)

    # Shuffle indices and split into chunks
    indices = np.random.permutation(len(full_train_dataset))
    split_indices = np.array_split(indices, NUM_CLIENTS)

    # Create one Subset for each client
    client_datasets = [Subset(full_train_dataset, idxs) for idxs in split_indices]
    return client_datasets
client_datasets = load_and_partition_cifar10()
PARTITION_DIR = "./partitions"

# loading the whole database
def prepare_and_save_partitions():
    transform = transforms.ToTensor()
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)

    indices = np.random.permutation(len(dataset))
    split_indices = np.array_split(indices, NUM_CLIENTS)

    # Save each client's indices
    for client_id, idxs in enumerate(split_indices):
        np.save(f"{PARTITION_DIR}/client_{client_id}.npy", idxs)
# Run this once before training
#prepare_and_save_partitions()


'''   Plot   
### Vis the training data
trainloader, _, _ = load_datasets(partition_id=0)
images, labels = batch["img"], batch["label"]
batch = next(iter(trainloader))
# Reshape and convert images to a NumPy array
# matplotlib requires images with the shape (height, width, 3)
images = images.permute(0, 2, 3, 1).numpy()

# Denormalize
images = images / 2 + 0.5

# Create a figure and a grid of subplots
fig, axs = plt.subplots(4, 8, figsize=(12, 6))

# Loop over the images and plot them
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
    ax.axis("off")

# Show the plot
fig.tight_layout()
#plt.show()
'''



### Our model
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
## Typical Train and validation functions
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()  # Set to read/write
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            #images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            images, labels = batch  # unpack the tuple
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval() #Set to read only
    with torch.no_grad():
        for batch in testloader:
            #images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            images, labels = batch  # unpack the tuple
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

## For sending the model peramiters (state dic, or gradiants) to and fro
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# Client, called from server?
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    # Returns the local model perams
    def get_parameters(self, config):
        return get_parameters(self.net)

    # recives permams from server 
    # train the model on local data
    # return updated model perams to server
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    # recives permams from server 
    # Evaluate the model on local data (validation)
    # Return the results to the server
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

## Non Fed Training
'''
trainloader, valloader, testloader = load_datasets(partition_id=0)
net = Net().to(DEVICE)

for epoch in range(5):
    train(net, trainloader, 1)
    loss, accuracy = test(net, valloader)
    print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

loss, accuracy = test(net, testloader)
print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
'''


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node

    partition_id = int(context.node_config["partition-id"])
    #trainloader, valloader, _ = load_datasets(partition_id=partition_id)
    # Load this client's dataset
    #client_dataset = client_datasets[partition_id]  
    #trainloader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #valloader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #del client_dataset
    partition_path = f"./partitions/client_{partition_id}.npy"
    indices = np.load(partition_path)

    # Create dataset and subset
    transform = transforms.ToTensor()
    dataset = CIFAR10(root="./data", train=True, download=False, transform=transform)
    subset = Subset(dataset, indices)
    del dataset

    # Create data loaders
    trainloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)  # use the same for now
    del subset


    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader).to_client()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=int(NUM_CLIENTS/2),  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available

    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training (epochs)
    config = ServerConfig(num_rounds=10)

    return ServerAppComponents(strategy=strategy, config=config)


server = ServerApp(server_fn=server_fn) # Create the ServerApp
client = ClientApp(client_fn=client_fn) # Create the ClientApp

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)