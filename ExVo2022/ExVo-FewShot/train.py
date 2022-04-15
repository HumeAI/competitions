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

from argparse import Namespace
from pathlib import Path
from losses import Losses
from torch.utils.data import DataLoader
from data_provider import get_dataloader
from model import Model
from metric_provider import MetricProvider
from collections import defaultdict


class Train:
    def __init__(self, dataset_params: dict, network_params: dict, train_params: dict):
        """Performs the training of the model.

        Args:
          dataset_params (dict): the parameters to load the daset.
              'class'      (str): The dataset class to provide.
              'train_path' (str): The path to the training csv file.
              'valid_path' (str): The path to the validation csv file.
              'batch_size' (int): Batch size to use during training.

          network_params (dict): the parameters to load the network. Depend on the network used.
                'network_type' (str)  : The network type to use.
                 *args, **kwargs.     : The parameters of the network.

          train_params (dict): Training parameters.
                'number_of_epochs' (int)  : Number of epochs to train model.
                'lr'               (float): learning rate of the model.
                'loss_name'        (str)  : The loss to use. Only one value 'cross_entropy_loss',
                'base_dir'         (str)  : The directory to save best model/logs.
                'eval_name'        (str)  : The evaluation metric to use. Only one value:'accuracy',
                'use_gpu'          (bool) : Whether to use GPU.
                'log_path'         (str)  : Path to save log file.
                'nepochs2stop'     (int)  : If model has not improved for `nepochs2stop`, then stop training.
        """

        self.train_dataset = get_dataloader(
            dataset_params["class"],
            dataset_params["batch_size"],
            shuffle=True,
            **dataset_params["train"],
        )

        self.valid_dataset = get_dataloader(
            dataset_params["class"], 1, shuffle=False, **dataset_params["valid"]
        )

        model_input_size = self.train_dataset.dataset.num_input_features
        self.model = Model(model_input_size, **network_params)

        self.train_params = Namespace(**train_params)

        self.loss_fn = Losses(self.train_params.loss_name).loss_fn

        self.metric2track = self.train_params.metric2track
        self.eval_fns = {
            eval_name: MetricProvider(eval_name)._metric
            for eval_name in self.train_params.eval_names
        }

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.train_params.lr)

        self.base_dir = Path(self.train_params.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if self.train_params.use_gpu:
            self.model.cuda()

        self.logger = self.get_logger(self.train_params.log_path)
        logging.info(self.model)

        if self.train_params.model_path:
            self.load_checkpoint(self.train_params.model_path)

    def load_checkpoint(self, ckpt_path):
        """Loads model parameters (state_dict) from file_path.
            If optimizer is provided, loads state_dict of
            optimizer assuming it is present in checkpoint.

        Args:
            checkpoint (str): Filename which needs to be loaded
            model (torch.nn.Module): Model for which the parameters are loaded
            optimizer (torch.optim): Optional: resume optimizer from checkpoint
        """

        logging.info("Restoring model from [{}]".format(str(ckpt_path)))

        if not Path(ckpt_path).exists():
            raise Exception("File doesn't exist [{}]".format(str(ckpt_path)))
        checkpoint = torch.load(str(ckpt_path))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return checkpoint

    def get_logger(self, log_path: str):
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
        logging.info("Starting Training")

        improved = 0
        best_score = float("-inf")

        self.model_dir = self.base_dir / "model"
        self.model_dir.mkdir(exist_ok=True)

        for epoch in range(self.train_params.number_of_epochs):
            logging.info(f"Epoch: {epoch+1}/{self.train_params.number_of_epochs}")

            train_score = self.perform_train_epoch()

            # Evaluate for one epoch on training and validation sets
            with torch.no_grad():
                valid_score = self.perform_eval_epoch(self.valid_dataset, "Validation")

            logging.info("")

            total_score = float(valid_score)
            if total_score > best_score:
                logging.info("New best score. Saving model...")
                improved = 0
                best_score = total_score
                self.save_model(True, epoch, train_score, total_score)

            improved += 1
            self.save_model(False, epoch, train_score, valid_score)

            if improved > self.train_params.nepochs2stop:
                break

        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    def save_model(self, best_model: bool, epoch, train_score, valid_score):
        model_name = "best_model" if best_model else "last_model"

        save_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "valid_score": valid_score,
            "train_score": train_score,
        }

        torch.save(save_dict, str(self.model_dir / f"{model_name}.pth.tar"))

        if best_model:
            np.savetxt(
                f"{self.train_params.base_dir}/best_score.txt",
                np.array([valid_score]),
                fmt="%f",
            )

    def perform_train_epoch(self):

        self.model.train()

        label_names = self.train_dataset.dataset.label_names
        loss_name = self.train_params.loss_name
        take_last_frame = self.train_params.take_last_frame

        eval_scores = {str(x): [] for x in label_names}
        total_iter, total_loss_score = 0, 0

        predictions = defaultdict(list)
        ground_truth = defaultdict(list)
        for n_iter, (batch_data, batch_gt, mask, f) in enumerate(self.train_dataset):
            print(f"Iter:{n_iter+1}/{len(self.train_dataset)}")

            self.optimizer.zero_grad()

            if self.train_params.use_gpu:
                batch_data = batch_data.cuda()
                batch_gt = batch_gt.cuda()

            batch_preds = self.model(batch_data)

            batch_loss_pred = 0
            for l, label_name in enumerate(label_names):
                pred_loss = self.loss_fn(
                    batch_preds[..., l], batch_gt[..., l], mask, take_last_frame
                )

                batch_loss_pred += pred_loss

            batch_loss_pred /= len(label_names)

            batch_loss = batch_loss_pred

            batch_loss.backward()
            self.optimizer.step()

            if n_iter % self.train_params.save_summary_steps == 0:
                total_loss_score += batch_loss
                eval_score, batch_loss = 0, 0

                for l, name in enumerate(label_names):
                    for i, m in enumerate(mask):
                        mframe = m - 1 if take_last_frame else range(m)
                        predictions[name].append(
                            batch_preds[i, mframe, l].reshape(
                                -1,
                            )
                        )
                        ground_truth[name].append(
                            batch_gt[i, mframe, l].reshape(
                                -1,
                            )
                        )

                total_iter += 1

        total_loss_score /= total_iter

        for l, name in enumerate(label_names):
            predictions[name] = torch.cat(predictions[name])
            ground_truth[name] = torch.cat(ground_truth[name])

        # Compute evaluation performance
        eval_performace = {
            name: defaultdict(float) for name, _ in self.eval_fns.items()
        }
        for l, label_name in enumerate(label_names):
            for eval_name, eval_fn in self.eval_fns.items():
                score_dim = eval_fn(predictions[label_name], ground_truth[label_name])
                eval_performace[eval_name][label_name] = score_dim

        # Compute mean scores
        mean_scores = {
            eval_name: torch.mean(
                torch.tensor([label_score for label_score in label_name.values()])
            )
            for eval_name, label_name in eval_performace.items()
        }

        # Print mean scores
        eval_scores_print = "Eval Performance: "
        for eval_name, mean_score in mean_scores.items():
            eval_scores_print += f"{eval_name}: {mean_score}  "

        loss_print = f"Loss ({self.train_params.loss_name}): {total_loss_score}"
        logging.info(f"Training Results: {loss_print} - {eval_scores_print}")

        print(f"Training Results: {loss_print} - {eval_scores_print}")
        score_to_return = mean_scores[self.metric2track]

        return np.array(score_to_return)

    def perform_eval_epoch(self, dataset: DataLoader, task: str):

        self.model.eval()

        label_names = self.train_dataset.dataset.label_names
        eval_scores = {str(x): [] for x in label_names}
        take_last_frame = self.train_params.take_last_frame

        total_iter = 0

        predictions = defaultdict(list)
        ground_truth = defaultdict(list)
        for n_iter, (batch_data, batch_gt, mask, _) in enumerate(dataset):
            print(f"Iter:{n_iter+1}/{len(dataset)}")

            if self.train_params.use_gpu:
                batch_data = batch_data.cuda()
                batch_gt = batch_gt.cuda()

            batch_preds = self.model(batch_data)

            for l, name in enumerate(label_names):
                for i, m in enumerate(mask):
                    mframe = m - 1 if take_last_frame else range(m)
                    predictions[name].append(
                        batch_preds[i, mframe, l].reshape(
                            -1,
                        )
                    )
                    ground_truth[name].append(
                        batch_gt[i, mframe, l].reshape(
                            -1,
                        )
                    )

            total_iter += 1

        # Concat predictions/gt
        for l, name in enumerate(label_names):
            predictions[name] = torch.cat(predictions[name])
            ground_truth[name] = torch.cat(ground_truth[name])

        # Compute evaluation performance per metric
        eval_performace = {
            name: defaultdict(float) for name, _ in self.eval_fns.items()
        }
        for l, label_name in enumerate(label_names):
            for eval_name, eval_fn in self.eval_fns.items():
                score_dim = eval_fn(predictions[label_name], ground_truth[label_name])
                eval_performace[eval_name][label_name] = score_dim

        # Compute mean scores
        mean_scores = {
            eval_name: torch.mean(
                torch.tensor([label_score for label_score in label_scores.values()])
            )
            for eval_name, label_scores in eval_performace.items()
        }

        # Print mean scores
        eval_scores_print = "Eval Performance: "
        for eval_name, mean_score in mean_scores.items():
            eval_scores_print += f"{eval_name}: {mean_score}  "

        print(f"Validation Results: {eval_scores_print}")
        logging.info(f"{task} Results: {eval_scores_print}")

        score_to_return = mean_scores[self.metric2track]

        return np.array(score_to_return)
