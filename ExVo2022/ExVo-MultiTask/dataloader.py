# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import csv
from pathlib import Path


class Dataloader:
    def create(
        labels,
        store_files,
        data_dir,
        feature_type,
        classes,
        age_class,
        country_classes,
        store_name,
        return_val=False
    ):
        print("No stored files found, creating from scratch ... ")
        train_X, val_X, test_X = [], [], []
        train_y, val_y, test_y = [], [], []
        train_age, val_age, test_age = [], [], []
        train_country, val_country, test_country = [], [], []
        val_filename = []
        test_filename = []

        tmp_folder = Path("tmp/")
        if not tmp_folder.is_file():
            tmp_folder.mkdir(exist_ok=True)

        feat_dict = {
            "ComParE": [",", "infer", 1, 6373],
            "eGeMAPS": [",", "infer", 1, 88],
            "DeepSpectrum": [",", "infer", 1, 4095],
            "openXBOW/125": [",", None, 1, 125],
            "openXBOW/250": [",", None, 1, 250],
            "openXBOW/500": [",", None, 1, 500],
            "openXBOW/1000": [",", None, 1, 1000],
            "openXBOW/2000": [",", None, 1, 2000],
        }

        sep_type = feat_dict[feature_type][0]
        header_type = feat_dict[feature_type][1]
        columns_rm = feat_dict[feature_type][2]
        feat_dimensions = feat_dict[feature_type][3]

        for filename in tqdm(glob(f"{data_dir}/feats/{feature_type}/*.csv")):
            file_id = filename.split("/")[-1][:-4]
            partition = labels["Split"][labels["File_ID"].str.contains(file_id)]
            lab_index = partition.index.values[0]
            partition = partition.values[0]
            with open(filename, "r") as csvfile:
                reader = csv.reader(csvfile)
                for index, row in enumerate(reader):
                    if sep_type != ",":
                        row = row[0].replace(sep_type, ",")
                    if index != 0:
                        if partition == "Train":
                            train_X.append(row[columns_rm:])
                        elif partition == "Val":
                            val_X.append(row[columns_rm:])
                        elif partition == "Test":
                            test_X.append(row[columns_rm:])
                    last_count = index
                df_shape = last_count
                label_id_high = [
                    labels.loc[lab_index:lab_index, classes].values[0]
                ] * df_shape
                label_id_age = [
                    labels.loc[lab_index:lab_index, age_class].values[0]
                ] * df_shape
                label_id_country = [
                    labels.loc[lab_index:lab_index, country_classes].values[0]
                ] * df_shape
                if partition == "Train":
                    train_y.append(label_id_high)
                    train_age.append(label_id_age)
                    train_country.append(label_id_country)
                elif partition == "Val":
                    val_y.append(label_id_high)
                    val_age.append(label_id_age)
                    val_country.append(label_id_country)
                    val_filename.append(file_id)
                elif partition == "Test":
                    test_y.append(label_id_high)
                    test_age.append(label_id_age)
                    test_country.append(label_id_country)
                    test_filename.append(file_id)

        train_X_group, val_X_group, test_X_group = (
            pd.DataFrame(train_X),
            pd.DataFrame(val_X),
            pd.DataFrame(test_X),
        )
        train_y_group, val_y_group, test_y_group = (
            pd.DataFrame(np.vstack(train_y)),
            pd.DataFrame(np.vstack(val_y)),
            pd.DataFrame(np.vstack(test_y)),
        )
        train_age_group, val_age_group, test_age_group = (
            pd.DataFrame(train_age),
            pd.DataFrame(val_age),
            pd.DataFrame(test_age),
        )
        train_country_group, val_country_group, test_country_group = (
            pd.DataFrame(train_country),
            pd.DataFrame(val_country),
            pd.DataFrame(test_country),
        )

        val_filename_group = pd.DataFrame(val_filename)
        test_filename_group = pd.DataFrame(test_filename)

        print("Saving data ...")
        if store_files:
            train_X_group.to_csv(f"tmp/{store_name}_train_X.csv", index=False)
            val_X_group.to_csv(f"tmp/{store_name}_val_X.csv", index=False)
            test_X_group.to_csv(f"tmp/{store_name}_test_X.csv", index=False)
            train_y_group.to_csv(f"tmp/{store_name}_train_y_high.csv", index=False)
            val_y_group.to_csv(f"tmp/{store_name}_val_y_high.csv", index=False)
            test_y_group.to_csv(f"tmp/{store_name}_test_y_high.csv", index=False)
            train_age_group.to_csv(f"tmp/{store_name}_train_y_age.csv", index=False)
            val_age_group.to_csv(f"tmp/{store_name}_val_y_age.csv", index=False)
            test_age_group.to_csv(f"tmp/{store_name}_test_y_age.csv", index=False)

            train_country_group.to_csv(
                f"tmp/{store_name}_train_y_country.csv", index=False
            )
            val_country_group.to_csv(f"tmp/{store_name}_val_y_country.csv", index=False)
            test_country_group.to_csv(
                f"tmp/{store_name}_test_y_country.csv", index=False
            )
            val_filename_group.to_csv(
                f"tmp/{store_name}_val_filename.csv", index=False
            )
            test_filename_group.to_csv(
                f"tmp/{store_name}_test_filename.csv", index=False
            )

        comb = [train_X_group, val_X_group, test_X_group]
        high = [train_y_group, val_y_group, test_y_group]
        age = [train_age_group, val_age_group, test_age_group]
        country = [train_country_group, val_country_group, test_country_group]

        if return_val:
            return comb, high, age, country, feat_dimensions, test_filename_group, val_filename_group
        else:
            return comb, high, age, country, feat_dimensions, test_filename_group

    def load(feature_type, store_name, return_val=False):
        feat_dict = {
            "ComParE": [";", "infer", 2, 6373],
            "eGeMAPS": [";", "infer", 2, 88],
            "DeepSpectrum": [",", "infer", 2, 4095],
            "openXBOW/125": [",", None, 1, 125],
            "openXBOW/250": [",", None, 1, 250],
            "openXBOW/500": [",", None, 1, 500],
            "openXBOW/1000": [",", None, 1, 1000],
            "openXBOW/2000": [",", None, 1, 2000],
        }

        sep_type = feat_dict[feature_type][0]
        header_type = feat_dict[feature_type][1]
        columns_rm = feat_dict[feature_type][2]
        feat_dimensions = feat_dict[feature_type][3]

        print("Loading data ...")
        train_X_group, val_X_group, test_X_group = (
            pd.read_csv(f"tmp/{store_name}_train_X.csv"),
            pd.read_csv(f"tmp/{store_name}_val_X.csv"),
            pd.read_csv(f"tmp/{store_name}_test_X.csv"),
        )
        train_y_group, val_y_group, test_y_group = (
            pd.read_csv(f"tmp/{store_name}_train_y_high.csv"),
            pd.read_csv(f"tmp/{store_name}_val_y_high.csv"),
            pd.read_csv(f"tmp/{store_name}_test_y_high.csv"),
        )
        train_age_group, val_age_group, test_age_group = (
            pd.read_csv(f"tmp/{store_name}_train_y_age.csv"),
            pd.read_csv(f"tmp/{store_name}_val_y_age.csv"),
            pd.read_csv(f"tmp/{store_name}_test_y_age.csv"),
        )
        train_country_group, val_country_group, test_country_group = (
            pd.read_csv(f"tmp/{store_name}_train_y_country.csv"),
            pd.read_csv(f"tmp/{store_name}_val_y_country.csv"),
            pd.read_csv(f"tmp/{store_name}_test_y_country.csv"),
        )
        val_filename_group = pd.read_csv(f"tmp/{store_name}_val_filename.csv")
        test_filename_group = pd.read_csv(f"tmp/{store_name}_test_filename.csv")

        comb = [train_X_group, val_X_group, test_X_group]
        high = [train_y_group, val_y_group, test_y_group]
        age = [train_age_group, val_age_group, test_age_group]
        country = [train_country_group, val_country_group, test_country_group]

        if return_val:
            return comb, high, age, country, feat_dimensions, test_filename_group, val_filename_group
        else:
            return comb, high, age, country, feat_dimensions, test_filename_group
