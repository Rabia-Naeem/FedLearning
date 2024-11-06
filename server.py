import argparse
import time
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict, Callable

import flwr as fl
import torch.nn.functional as F
import torch_geometric.loader.dataloader
import crypten
from flwr.common import FitRes, Parameters, Scalar, NDArrays, FitIns, EvaluateIns, MetricsAggregationFn, \
    ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

# Initialize Crypten for server with world_size=5 and rank=0
world_size = 5
rank = 0
crypten.init_thread(rank=rank, world_size=world_size)

import common
from utils import *

BATCH_SIZE, TEST_BATCH_SIZE = 512, 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def secure_aggregate_parameters(client_params):
    encrypted_params = [crypten.cryptensor(param) for param in client_params]
    encrypted_agg = encrypted_params[0]
    for param in encrypted_params[1:]:
        encrypted_agg += param
    agg_params = encrypted_agg.get_plain_text()
    return agg_params / len(client_params)

# Define evaluation function
def get_eval_fn(model) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    test_data = TestbedDataset(root=FOLDER, dataset='kiba' + '_test')
    test_loader = torch_geometric.loader.dataloader.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, float]]:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        loss_mse = 0
        print(f'Make prediction for {len(test_loader.dataset)} samples...')

        with torch.no_grad():
            for data in test_loader:
                data, target = data.to(DEVICE, non_blocking=True), data.y.view(-1, 1).float().to(DEVICE, non_blocking=True)
                output = model(data)
                loss_mse += F.mse_loss(output, target, reduction="sum")

            mse = float(loss_mse / len(test_loader.dataset))
        return mse, {'MSE': mse}

    return evaluate

class SaveModelStrategy(FedAvg):
    EARLY_STOP = False

    def __init__(self, *args, early_stopping_epochs=5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.early_stopping = False
        self.epochs_without_improvement = 0
        self.last_better_loss_value = 1000
        self.early_stopping_epochs = early_stopping_epochs
        self.best_model = None
        self.best_metric = None

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        client_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        aggregated_weights = secure_aggregate_parameters(client_params)
        return ndarrays_to_parameters(aggregated_weights)

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        loss, metrics = super().evaluate(server_round, parameters)
        if self.early_stopping_epochs >= 0:
            if loss < self.last_better_loss_value:
                self.last_better_loss_value = loss
                self.epochs_without_improvement = 0
                self.best_model = parameters
                self.best_metric = metrics
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement > self.early_stopping_epochs:
                    self.early_stopping = True
                    print("EARLY STOPPING TRIGGERED")
                    weights = parameters_to_ndarrays(self.best_model)
                    np.savez(f"{args.save_name}.npz", *weights)
                    loss = self.last_better_loss_value
                    metrics = self.best_metric
        return loss, metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        fit_list = super().configure_fit(server_round, parameters, client_manager)
        return [] if self.early_stopping else fit_list

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
        return [] if self.early_stopping else evaluate_list

def main(args):
    model = common.create_model(args.normalisation, DEVICE)
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_fn=get_eval_fn(model),
        initial_parameters=ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()]),
        early_stopping_epochs=args.early_stop
    )

    fl.server.start_server(strategy=strategy, config=fl.server.ServerConfig(num_rounds=args.num_rounds))

start_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-clients", default=2, type=int)
    parser.add_argument("--num-rounds", default=1, type=int)
    parser.add_argument("--early-stop", default=-1, type=int)
    parser.add_argument("--folder", default='data/', type=str)
    parser.add_argument("--seed", type=int, required=True, help="Seed for data partitioning")
    parser.add_argument("--diffusion", action='store_true')
    parser.add_argument("--diffusion-folder", default=None, type=str)
    parser.add_argument("--save-name", default=None, type=str)
    parser.add_argument("--normalisation", default="bn", type=str)
    args = parser.parse_args()

    global NUM_CLIENTS, SEED, DIFFUSION, FOLDER, DIFFUSION_FOLDER, NORMALISATION
    NUM_CLIENTS = args.num_clients
    SEED = args.seed
    DIFFUSION = args.diffusion
    FOLDER = args.folder
    DIFFUSION_FOLDER = args.diffusion_folder
    NORMALISATION = args.normalisation

    main(args)
print("--- %s seconds ---" % (time.time() - start_time))
