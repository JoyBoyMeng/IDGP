import os
import torch
import torch.nn as nn
import logging


class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str, logger: logging.Logger, model_name: str = None):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param save_model_name: str, save model name
        :param logger: Logger
        :param model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.logger = logger
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
        self.save_model_name = save_model_name
        self.save_model_folder = save_model_folder
        self.model_name = model_name
        if self.model_name in ['JODIE', 'DyRep', 'TGN', 'TGN-id', 'Fluxion']:
            # path to additionally save the nonparametric data (e.g., tensors) in memory-based models (e.g., JODIE, DyRep, TGN)
            self.save_model_nonparametric_data_path = os.path.join(save_model_folder, f"{save_model_name}_nonparametric_data.pkl")
            self.save_model_nonparametric_data_path1 = os.path.join(save_model_folder,
                                                                   f"{save_model_name}_item_flux_hist.pkl")
            self.save_model_nonparametric_data_path2 = os.path.join(save_model_folder,
                                                                   f"{save_model_name}_item_time_hist.pkl")

    def step(self, metrics: list, model: nn.Module, epochs: str):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: nn.Module
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_model(model)
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        self.save_checkpoint(model, epochs)
        return self.early_stop

    def save_model(self, model: nn.Module):
        """
        saves model at self.save_model_path
        :param model: nn.Module
        :return:
        """
        self.logger.info(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)
        if self.model_name in ['JODIE', 'DyRep', 'TGN', 'TGN-id', 'Fluxion']:
            torch.save(model[0].memory_bank.node_raw_messages, self.save_model_nonparametric_data_path)
            torch.save(model[0].item_flux_hist, self.save_model_nonparametric_data_path1)
            torch.save(model[0].item_time_hist, self.save_model_nonparametric_data_path2)

    def save_checkpoint(self, model: nn.Module, epochs: str):
        """
        saves checkpoint at save_checkpoint_path
        :param model: nn.Module
        :return:
        """
        save_checkpoint_folder = os.path.join(self.save_model_folder, f'epochs{epochs}')
        os.makedirs(save_checkpoint_folder, exist_ok=True)
        save_checkpoint_path = os.path.join(save_checkpoint_folder, f"{self.save_model_name}.pkl")
        self.logger.info(f"save model {save_checkpoint_path}")
        torch.save(model.state_dict(), save_checkpoint_path)
        if self.model_name in ['JODIE', 'DyRep', 'TGN', 'TGN-id', 'Fluxion']:
            save_model_nonparametric_data_path = os.path.join(save_checkpoint_folder,
                                                                   f"{self.save_model_name}_nonparametric_data.pkl")
            save_model_nonparametric_data_path1 = os.path.join(save_checkpoint_folder,
                                                                    f"{self.save_model_name}_item_flux_hist.pkl")
            save_model_nonparametric_data_path2 = os.path.join(save_checkpoint_folder,
                                                                    f"{self.save_model_name}_item_time_hist.pkl")
            torch.save(model[0].memory_bank.node_raw_messages, save_model_nonparametric_data_path)
            torch.save(model[0].item_flux_hist, save_model_nonparametric_data_path1)
            torch.save(model[0].item_time_hist, save_model_nonparametric_data_path2)


    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        self.logger.info(f"load model {self.save_model_path}")
        model.load_state_dict(torch.load(self.save_model_path, map_location=map_location))
        if self.model_name in ['JODIE', 'DyRep', 'TGN', 'TGN-id', 'Fluxion']:
            model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path,
                                                                map_location=map_location,
                                                                weights_only=False)
            model[0].item_flux_hist = torch.load(self.save_model_nonparametric_data_path1,
                                                                map_location=map_location,
                                                                weights_only=False)
            model[0].item_time_hist = torch.load(self.save_model_nonparametric_data_path2,
                                                 map_location=map_location,
                                                 weights_only=False)

    def load_checkpoint_4test(self, model: nn.Module, map_location: str = None, epochs: str=''):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        save_checkpoint_folder = os.path.join(self.save_model_folder, f'epochs{epochs}')
        save_checkpoint_path = os.path.join(save_checkpoint_folder, f"{self.save_model_name}.pkl")
        self.logger.info(f"load model {save_checkpoint_path}")
        model.load_state_dict(torch.load(save_checkpoint_path, map_location=map_location))
        if self.model_name in ['JODIE', 'DyRep', 'TGN', 'TGN-id', 'Fluxion']:
            save_model_nonparametric_data_path = os.path.join(save_checkpoint_folder,
                                                              f"{self.save_model_name}_nonparametric_data.pkl")
            save_model_nonparametric_data_path1 = os.path.join(save_checkpoint_folder,
                                                               f"{self.save_model_name}_item_flux_hist.pkl")
            save_model_nonparametric_data_path2 = os.path.join(save_checkpoint_folder,
                                                               f"{self.save_model_name}_item_time_hist.pkl")
            model[0].memory_bank.node_raw_messages = torch.load(save_model_nonparametric_data_path,
                                                                map_location=map_location,
                                                                weights_only=False)
            model[0].item_flux_hist = torch.load(save_model_nonparametric_data_path1,
                                                                map_location=map_location,
                                                                weights_only=False)
            model[0].item_time_hist = torch.load(save_model_nonparametric_data_path2,
                                                 map_location=map_location,
                                                 weights_only=False)

    def load_checkpoint_4visualization(self, model: nn.Module, map_location: str = None, epochs: str=''):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        if epochs != '':
            save_checkpoint_folder = os.path.join(self.save_model_folder, f'epochs{epochs}')
        else:
            save_checkpoint_folder = self.save_model_folder
        save_checkpoint_path = os.path.join(save_checkpoint_folder, f"{self.save_model_name}.pkl")
        self.logger.info(f"load model {save_checkpoint_path}")
        model.load_state_dict(torch.load(save_checkpoint_path, map_location=map_location))
