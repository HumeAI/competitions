# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch.optim as optim
import torch
import torch.nn as nn
import logging
import numpy as np
import json

from argparse import Namespace
from pathlib import Path
from losses import Losses
from torch.utils.data import DataLoader
from data_provider import get_dataloader
from model import Model
from metric_provider import MetricProvider
from collections import defaultdict


class Evaluation:
    def __init__(self, dataset_params, network_params, params):
        """Performs the evaluation of the model.

        Args:
          dataset_params (dict): the parameters to load the daset.
              'class'      (str): The dataset class to provide.
              'test'       (str): The path to the test csv file.
              'batch_size' (int): Batch size to use during training.

          network_params (dict): the parameters to load the network. Depend on the network used.

          params (dict): Training parameters.
                'model_path'       (str)  : The directory to save best model/logs.
                'eval_name'        (str)  : The evaluation metric to use. Only one value:'accuracy',
                'use_gpu'          (bool) : Whether to use GPU.
                'log_path'         (str)  : Path to save log file.
        """

        self.dataset = get_dataloader(
            dataset_params["class"],
            dataset_params["batch_size"],
            shuffle=False,
            **dataset_params["test"],
        )

        model_input_size = self.dataset.dataset.num_input_features
        self.model = Model(model_input_size, **network_params)

        self.params = Namespace(**params)
        self.eval_fns = {
            eval_name: MetricProvider(eval_name)._metric
            for eval_name in self.params.eval_names
        }
        self.loss_fn = Losses(self.params.loss_name).loss_fn
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.params.lr)
        self.label_names = self.dataset.dataset.label_names

        self.n_support = self.dataset.dataset.n_support
        self.n_query = self.dataset.dataset.n_query

        if self.params.use_gpu:
            self.model.cuda()

        self.logger = self.get_logger(self.params.log_path)
        logging.info(self.model)

        self.load_model(self.params.use_gpu)
        self._params_to_train(self.model)

    def _params_to_train(self, model):
        for i, layer in enumerate(model.children()):
            for param in layer.parameters():
                param.requires_grad = False

        for param in model.rnn.parameters():
            param.requires_grad = True

        for param in model.linear.parameters():
            param.requires_grad = True

    def load_model(self, use_gpu):
        """Loads model parameters (state_dict) from file_path."""

        device = torch.device("cpu") if not use_gpu else torch.device("cuda:0")
        logging.info(f"Loading model parameter from {str(self.params.model_path)}")

        if not Path(self.params.model_path).exists():
            raise Exception(f"No model exists in path : {str(self.params.model_path)}")
        checkpoint = torch.load(str(self.params.model_path), map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        return checkpoint

    def get_logger(self, log_path):
        """Creates the logger to log output.

        Args:
          log_path (str): Path to save log file.
        """

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)-4s %(levelname)-4s: %(message)s",
            datefmt="%d-%m-%y %H:%M",
            filename=log_path,
            filemode="w",
        )

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # add the handler to the root logger
        logging.getLogger("").addHandler(console)

    def start(self):
        logging.info("Starting Evaluation")

        self.perform_few_shot(self.dataset)

        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    def save_predictions(self, predictions):

        save_preds_path = Path(self.params.base_dir) / "predictions.json"

        with open(str(save_preds_path), "w") as f:
            json.dump(predictions, f, indent=2)

    def perform_few_shot(self, dataset):
        self.model.train()

        loss_name = self.params.loss_name
        take_last_frame = self.params.take_last_frame

        eval_scores = {str(x): [] for x in self.label_names}
        predictions = defaultdict(list)
        for n_iter, (batch_data, batch_gt, mask, filenames) in enumerate(dataset):
            print(f"Iter:{n_iter+1}/{len(dataset)}")

            self.optimizer.zero_grad()

            if self.params.use_gpu:
                batch_data = batch_data.cuda()
                batch_gt = batch_gt.cuda()

            batch_preds = self.model(batch_data)
            batch_support = self._get_support_query(batch_preds, mask, batch_gt)

            self.train_model(batch_support)
            batch_preds_query = self.model(batch_data[-self.n_query :, ...])

            query_filenames = filenames[-self.n_query :]
            query_mask = mask[-self.n_query :]
            for l, name in enumerate(query_filenames):
                for i, m in enumerate(query_mask):
                    mframe = m - 1 if take_last_frame else range(m)
                    predictions[Path(name).name].append(
                        batch_preds[i, mframe, :]
                        .reshape(
                            -1,
                        )
                        .data.cpu()
                        .numpy()
                        .astype(np.float32)
                        .tolist()
                    )

        self.save_predictions(predictions)

    def train_model(self, batch_support):
        batch_preds, batch_gt = batch_support[0], batch_support[1]

        batch_loss = 0
        for l, label_name in enumerate(self.label_names):
            pred_loss = self.loss_fn(batch_preds[..., l], batch_gt[..., l])
            batch_loss += pred_loss

        batch_loss /= len(self.label_names)

        batch_loss.backward()
        self.optimizer.step()

        return batch_loss.item()

    def _get_support_query(self, batch_preds, masks, batch_gt):
        batch_size = len(batch_preds)

        n_support = self.n_support
        n_query = self.n_query
        num_subjects = batch_size // (n_support + n_query)

        total_samples_subject = n_support + n_query
        support_preds, support_gt = [], []

        for i in range(num_subjects):
            ss = i * total_samples_subject

            sup_inds = list(range(ss, ss + n_support))
            for s in sup_inds:
                m = masks[s]
                support_preds.append(batch_preds[s, m - 1, :])
                support_gt.append(batch_gt[s, m - 1, :])

        return [torch.stack(support_preds), torch.stack(support_gt)]
