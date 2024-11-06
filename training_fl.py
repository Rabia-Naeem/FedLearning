import argparse
import time
from collections import OrderedDict
from typing import List
import os
import flwr as fl
import torch.nn.functional as F
import torch_geometric.loader.dataloader
import crypten
from numpy import ndarray
import common
from utils import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Client Script for Federated Learning with Crypten")
parser.add_argument("--server", type=str, required=True, help="Server address")
parser.add_argument("--rank", type=int, required=True, help="Unique rank for this client")
parser.add_argument("--world-size", type=int, default=5, help="Total number of participants (world size)")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--early-stop", type=int, default=-1)
parser.add_argument("--normalisation", type=str, default="ln")
parser.add_argument("--num-clients", type=int, default=4)
parser.add_argument("--partition", type=int, required=True)
parser.add_argument("--diffusion", action='store_true')
parser.add_argument("--diffusion-folder", type=str, required=True)
args = parser.parse_args()

# Initialize Crypten with specified rank and world size
crypten.init_thread(rank=args.rank, world_size=args.world_size)
print(f"Client started with rank {args.rank} and world size {args.world_size}")

# Set constants and device
BATCH_SIZE, TEST_BATCH_SIZE = 512, 512
LR = 0.0001
LOG_INTERVAL = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Flower client
class FedDTIClient(fl.client.NumPyClient):
    def __init__(self, model, train, test, cid):
        print(f"Initializing client {cid} with data sizes: train={len(train)}, test={len(test)}")
        self.device = DEVICE
        self.model = model.to(self.device, non_blocking=True)
        self.batch_size = BATCH_SIZE if len(train) > BATCH_SIZE and len(test) > BATCH_SIZE else min(len(train), len(test))
        self.train_loader = torch_geometric.loader.dataloader.DataLoader(train, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = torch_geometric.loader.dataloader.DataLoader(test, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        self.id = cid
        print("Client initialization complete!")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print('Training on {} samples...'.format(len(self.train_loader.dataset)))
        self.model.train()
        epoch = -1
        for batch_idx, data in enumerate(self.train_loader):
            data, target = data.to(self.device, non_blocking=True), data.y.view(-1, 1).float().to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(data), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print(f'Train epoch: {epoch} [{batch_idx}/{len(self.train_loader)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        print("Training complete!")
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def get_parameters(self, **kwargs) -> List[ndarray]:
        # Encrypt parameters before sending to server
        return [crypten.cryptensor(val.cpu().numpy()) for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        print("Setting model parameters...")
        # Decrypt received parameters
        decrypted_params = [param.get_plain_text() for param in parameters]
        params_dict = zip(self.model.state_dict().keys(), decrypted_params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_mse = 0
        print(f'Making predictions on {len(self.test_loader.dataset)} samples...')
        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                data, target = data.to(self.device, non_blocking=True), data.y.view(-1, 1).float().to(self.device, non_blocking=True)
                output = self.model(data)
                loss_mse += F.mse_loss(output, target, reduction="sum")
        loss = float(loss_mse / len(self.test_loader.dataset))
        print(f"Evaluation complete, MSE: {loss}")
        return loss, len(self.test_loader.dataset), {"mse": loss}


def main(args):
    print("Loading model...")
    model = common.create_model(args.normalisation)
    if not args.diffusion:
        train, test = common.load(args.num_clients, args.seed)[args.partition]
    else:
        print(f'Loading data from path: {os.path.join(args.folder, args.diffusion_folder, "client_" + str(args.partition))}')
        train, test = common.load(args.num_clients, args.seed, path=os.path.join(args.folder, args.diffusion_folder, 'client_' + str(args.partition)))

    print("Checking data integrity...")
    if train is None or test is None:
        print("Data loading failed!")
        return
    else:
        print(f"Train data: {len(train)} samples, Test data: {len(test)} samples")

    print(f"Starting client with server address {args.server}")
    client = FedDTIClient(model, train, test, args.partition)
    print("Starting Flower client connection...")
    fl.client.start_client(server_address=args.server, client=client)
    print("Client connection started...")


start_time = time.time()
if __name__ == "__main__":
    main(args)
print("--- %s seconds ---" % (time.time() - start_time))
