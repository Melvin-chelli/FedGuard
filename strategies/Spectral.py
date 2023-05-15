from typing import Union, Dict, List, Optional, Tuple

import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import torch
from torch.utils.tensorboard import SummaryWriter

from .aggregate import aggregate_spectral
from .TensorboardStrategy import TensorboardStrategy
from utils.models import VAE

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

class Spectral(TensorboardStrategy):
    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        fraction_fit,
        fraction_evaluate,
        eval_fn,
        writer,
        on_fit_config_fn,
        flat_model_shape,
        vae_model='/app/artifacts/vae_50_epochs.pt',
        ):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        eval_fn=eval_fn,
                        on_fit_config_fn=on_fit_config_fn,
                        writer=writer)
        
        self.writer = writer
        self.vae = VAE(input_dim=flat_model_shape).to(DEVICE)
        self.vae.load_state_dict(torch.load(vae_model))

    def __repr__(self) -> str:
        return "Spectral"


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using median."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        parameters_aggregated = ndarrays_to_parameters(
            aggregate_spectral(weights_results, self.vae, device=DEVICE)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated