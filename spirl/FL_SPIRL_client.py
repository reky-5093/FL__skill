from collections import OrderedDict
import os
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
import matplotlib; matplotlib.use('Agg')
import torch
from shutil import copy
import numpy as np
import Custome_train
from spirl.components.params import get_args
from spirl.utils.general_utils import AttrDict

WANDB_PROJECT_NAME = 'FL'
WANDB_ENTITY_NAME = 'yskang'
REWARD = 1
TMP = 100
NUM_CLIENTS = 7
NUM_ROUNDS = int(5e3)
GLOBAL_STEP = [0,0,0,0,0,0,0]


class SPIRLClient(fl.client.NumPyClient):
    def __init__(self, cid, args):
        self.cid = int(cid)
        self.model = Custome_train.ModelTrainer(args=args,cid=cid)
        self.model.global_step = GLOBAL_STEP[self.cid]

    def get_parameters(self,config):
        return [val.cpu().numpy() for _, val in self.model.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del( state_dict[d] )
        # parameters update
        self.model.model.load_state_dict(state_dict,strict=True)

    def fit(self, parameters, config):
        print("=============[fitting start]================") # 각 round를 구분하기위한 출력
        self.set_parameters(parameters)
        self.model.train()
        GLOBAL_STEP[self.cid] = self.model.global_step
        return self.get_parameters(config), TMP , {'round' : 1}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.model.val()
        return float(loss), TMP , {"Reward": float(REWARD)}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) :

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if (aggregated_parameters is not None) and (server_round%10 ==0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            os.makedirs("/workspace/skill/experiments/skill_prior_learning/kitchen/7_client_1/weights", exist_ok=True)
            np.savez(f"/workspace/skill/experiments/skill_prior_learning/kitchen/7_client_1/weights/round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def client_fn(cid) -> SPIRLClient:
    config = AttrDict(
        path = "spirl/configs/skill_prior_learning/kitchen/FL_hierarchial_cl",
        prefix = "3_client_{}".format(cid),
        new_dir = False,
        dont_save = False,
        resume = "",
        train = True,
        test_prediction = True,
        skip_first_val = False,
        val_sweep = False,
        gpu = -1,
        strict_weight_loading = True,
        deterministic = False,
        log_interval = 500,
        per_epoch_img_logs = 1,
        val_data_size = 160,
        val_interval = 5,
        detect_anomaly = False,
        feed_random_data = False,
        train_loop_pdb = False,
        debug = False,
        save2mp4 = False
    )
    return SPIRLClient(cid = cid, args = config)


if __name__ == "__main__":

    config = AttrDict(
        path = "spirl/configs/skill_prior_learning/kitchen/hierarchical_cl",
        prefix = "server_loss8",

        new_dir = False,
        dont_save = False,
        resume = "",
        train = True,
        test_prediction = True,
        skip_first_val = False,
        val_sweep = False,
        gpu = -1,
        strict_weight_loading = True,
        deterministic = False,
        log_interval = 500,
        per_epoch_img_logs = 1,
        val_data_size = 160,
        val_interval = 5,
        detect_anomaly = False,
        feed_random_data = False,
        train_loop_pdb = False,
        debug = False,
        save2mp4 = False
    )
    
    init_model =  Custome_train.ModelTrainer(args=config,cid=0)

    def evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        params_dict = zip(init_model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del( state_dict[d] )
        # parameters update
        init_model.global_step += 1
        init_model.model.load_state_dict(state_dict,strict=True)
        loss = init_model.val()
        return loss, {"accuracy": 1}



    """Create model, Create env, define Flower client, start Flower client."""
    strategy = SaveModelStrategy(
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=evaluate,
    #initial_parameters=fl.common.ndarrays_to_parameters(init_params),
)
    fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),  # Just three rounds
    strategy=strategy,
)
