from typing import Union, Dict, List, Optional, Tuple

import flwr as fl
import psutil
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from torch.utils.tensorboard import SummaryWriter


class TensorboardStrategy(fl.server.strategy.FedAvg):
    def __repr__(self) -> str:
        return "TensorboardStrategy"

    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        fraction_fit,
        fraction_evaluate,
        eval_fn,
        writer,
        on_fit_config_fn):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        evaluate_fn=eval_fn,
                        on_fit_config_fn=on_fit_config_fn)
        
        self.writer = writer
        self.bytes_recv_init_counter = psutil.net_io_counters().bytes_recv
        self.bytes_sent_init_counter = psutil.net_io_counters().bytes_sent

    def configure_fit(
        self, server_round, parameters, client_manager
    ):
        """Configure the next round of training."""
        clients_conf = super().configure_fit(server_round, parameters, client_manager)

        self.writer.add_scalar("Training/total_num_clients", len(clients_conf), server_round)

        # Return client/config pairs
        return clients_conf

    def evaluate(self, server_round, parameters):
        """Evaluate model parameters using an evaluation function."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Write scalars
        self.writer.add_scalar("Training/test_c_loss", loss, server_round)
        self.writer.add_scalar("Training/test_accuracy", metrics["accuracy"], server_round)
        self.writer.add_scalar("System/bytes_rcv", (psutil.net_io_counters().bytes_recv - self.bytes_recv_init_counter) / 1000000, server_round)
        self.writer.add_scalar("System/bytes_sent", (psutil.net_io_counters().bytes_sent - self.bytes_sent_init_counter) / 1000000, server_round)

        return loss, metrics
