import argparse
import time
from collections import OrderedDict
from typing import List
import os
import flwr as fl
import torch.nn.functional as F
import torch_geometric.loader.dataloader
from tqdm import tqdm  # For progress bar

from numpy import ndarray

import common
from utils import *
import crypten  # Import CrypTen for MPC
import crypten.nn as cnn  # Use CrypTen's neural networks

BATCH_SIZE, TEST_BATCH_SIZE = 512, 512
LR = 0.0001
LOG_INTERVAL = 20

# Step 1: Initialize CrypTen
crypten.init_thread(0, 2)

# Define Flower client
class FedDTIClient(fl.client.NumPyClient):

    def __init__(self, model, train, test, cid):
        print(f"Initializing client {cid} with data sizes: train={len(train)}, test={len(test)}")  # Debugging step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device, non_blocking=True)
        self.batch_size = BATCH_SIZE if len(train) > BATCH_SIZE and len(test) > BATCH_SIZE else min(len(train), len(test))
        self.train_loader = torch_geometric.loader.dataloader.DataLoader(train, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = torch_geometric.loader.dataloader.DataLoader(test, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        self.id = cid

        print("Client initialization complete!")  # Debugging step

    def fit(self, parameters, config):
        round_num = config.get("round", 0)  # Get the current round number from config
        print(f"Client {self.id} starting round {round_num}")  # Log the round information

        self.set_parameters(parameters)
        print(f'Training on {len(self.train_loader.dataset)} samples...')  # Debugging step
        self.model.train()

        # Start timing the training process
        start_time = time.time()

        # Encrypt the model for MPC
        crypten_model = cnn.from_pytorch(self.model, dummy_input=torch.empty((1, 10))).encrypt()

        # Use tqdm for a progress bar in the training loop
        for batch_idx, data in enumerate(tqdm(self.train_loader, desc=f"Client {self.id} Training", unit="batch")):
            data.x = crypten.cryptensor(data.x)  # Encrypt input data
            data.y = crypten.cryptensor(data.y.view(-1, 1).float())  # Encrypt target data

            crypten_model.zero_grad()
            output = crypten_model(data.x)
            loss = crypten.mse_loss(output, data.y)  # Compute encrypted MSE loss
            loss.backward()  # Backpropagate on encrypted data
            crypten_model.update_gradients(self.optimizer)  # Update gradients securely

            if batch_idx % LOG_INTERVAL == 0:
                print(f'Train batch: {batch_idx} Loss: {loss.get_plain_text().item():.6f}')  # Decrypt loss to print it

        # End timing the training process
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training complete! Time taken for training: {total_time:.2f} seconds.")  # Debugging step

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def get_parameters(self, **kwargs) -> List[ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        print("Setting model parameters...")  # Debugging step
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        round_num = config.get("round", 0)  # Get the current round number from config
        print(f"Client {self.id} starting evaluation for round {round_num}")  # Log the round information

        self.set_parameters(parameters)
        self.model.eval()
        loss_mse = 0

        # Start timing the evaluation process
        start_time = time.time()

        # Encrypt the model for evaluation
        crypten_model = cnn.from_pytorch(self.model, dummy_input=torch.empty((1, 10))).encrypt()

        print(f'Making predictions on {len(self.test_loader.dataset)} samples...')  # Debugging step
        with torch.no_grad():
            for _, data in enumerate(tqdm(self.test_loader, desc=f"Client {self.id} Evaluating", unit="batch")):
                data.x = crypten.cryptensor(data.x)  # Encrypt input data
                data.y = crypten.cryptensor(data.y.view(-1, 1).float())  # Encrypt target data

                output = crypten_model(data.x)
                loss_mse += crypten.mse_loss(output, data.y, reduction="sum").get_plain_text()  # Get decrypted loss

        # End timing the evaluation process
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Evaluation complete! MSE: {loss_mse:.6f} Time taken for evaluation: {total_time:.2f} seconds.")  # Debugging step

        loss = float(loss_mse / len(self.test_loader.dataset))
        return loss, len(self.test_loader.dataset), {"mse": loss}


def main(args):
    print("Loading model...")  # Debugging step
    model = common.create_model(NORMALISATION)
    if not DIFFUSION:
        train, test = common.load(NUM_CLIENTS, SEED)[args.partition]
    else:
        print(f'Loading data from path: {os.path.join(FOLDER, DIFFUSION_FOLDER, "client_" + str(args.partition))}')  # Debugging step
        train, test = common.load(NUM_CLIENTS, SEED, path=os.path.join(FOLDER, DIFFUSION_FOLDER, 'client_' + str(args.partition)))

    # Step 2: Check Data Integrity
    print("Checking data integrity...")  # Debugging step
    if train is None or test is None:
        print("Data loading failed!")  # Debugging step
        return
    else:
        print(f"Train data: {len(train)} samples, Test data: {len(test)} samples")  # Debugging step

    # Step 4: Increase Verbosity
    print(f"Starting client with server address {args.server}")  # Debugging step

    # Start Flower client
    client = FedDTIClient(model, train, test, args.partition)
    print("Starting Flower client connection...")  # Add before the start_client line
    fl.client.start_client(server_address=args.server, client=client)
    print("Client connection started...")  # Add right after to check if this is reached


# Step 5: Timeouts and Client Connections
start_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client Script")
    parser.add_argument("--num-clients", default=2, type=int)
    parser.add_argument("--num-rounds", default=1, type=int)
    parser.add_argument("--early-stop", default=-1, type=int)
    parser.add_argument("--folder", default=None, type=str)
    parser.add_argument("--seed", type=int, required=True, help="Seed for data partitioning")
    parser.add_argument("--diffusion", action='store_true')
    parser.add_argument("--diffusion-folder", default=None, type=str)
    parser.add_argument("--save-name", default=None, type=str)
    parser.add_argument("--normalisation", default="bn", type=str)
    parser.add_argument(
        "--partition", type=int, help="Data Partition to train on. Must be less than number of clients",
    )
    parser.add_argument(
        "--server", default='localhost:8080', type=str, help="server address", required=True,
    )
    args = parser.parse_args()

    global NUM_CLIENTS
    global SEED
    global DIFFUSION
    global FOLDER
    global DIFFUSION_FOLDER
    global NORMALISATION
    global DEVICE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLIENTS = args.num_clients
    SEED = args.seed
    DIFFUSION = args.diffusion
    FOLDER = args.folder
    DIFFUSION_FOLDER = args.diffusion_folder
    NORMALISATION = args.normalisation

    main(args)

# Print the total execution time for the entire client process
total_execution_time = time.time() - start_time
print(f"Total execution time for client: {total_execution_time:.2f} seconds.")