# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import csv
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class Dataloader:
    def create(
        store_name,
        task,
        data_dir,
        feature_type,
        labels,
        classes,
        sep_type,
        columns_rm,
    ):

        if not Path("tmp/").is_file():
            Path("tmp/").mkdir(exist_ok=True)

        splits_x = [[], [], []]
        splits_y = [[], [], []]
        filenames = [[], []]
        partitions = ["train", "val", "test"]

        stored_files = Path(f"tmp/{store_name}_train_y_{task}.csv")
        if not stored_files.is_file():
            for i, part in enumerate(partitions):
                list_files, clean_files = (
                    labels["File_ID"][labels["Split"] == part.capitalize()],
                    [],
                )
                for fileid in list_files:
                    clean_files.append(
                        f"{data_dir}/features/{feature_type}/{fileid[1:-1]}.csv"
                    )
                for filename in tqdm(clean_files[:]):
                    label_id = labels.loc[
                        labels["Split"][
                            labels["File_ID"].str.contains(filename.split("/")[-1][:-4])
                        ].index.values[0],
                        classes,
                    ]
                    with open(filename, "r") as csvfile:
                        reader = csv.reader(csvfile)
                        for index, row in enumerate(reader):
                            if sep_type != ",":
                                row = row[0].replace(sep_type, ",")
                        if index != 0:
                            splits_x[i].append(row[columns_rm:])
                            splits_y[i].append(label_id)
                            if i != 0:
                                filenames[i - 1].append(filename.split("/")[-1][:-4])

            stack_x = [[], [], []]
            stack_y = [[], [], []]
            stack_filenames = [[], []]

            for i, part in enumerate(partitions):
                stack_x[i] = pd.DataFrame(splits_x[i])
                stack_y[i] = pd.DataFrame(splits_y[i])

                if i != 0:
                    stack_filenames[i - 1] = pd.DataFrame(filenames[i - 1])

            print("Saving features ...")
            for i, part in enumerate(partitions):
                stack_x[i].to_csv(f"tmp/{store_name}_{part}_X.csv", index=False)
                stack_y[i].to_csv(f"tmp/{store_name}_{part}_y_{task}.csv", index=False)

                if i != 0:
                    stack_filenames[i - 1].to_csv(
                        f"tmp/{store_name}_{part}_filename.csv", index=False
                    )

        print("Loading features ...")
        X = [[], [], []]
        y = [[], [], []]
        stack_filenames = [[], []]
        for i, part in enumerate(partitions):
            X[i] = pd.read_csv(f"tmp/{store_name}_{part}_X.csv")
            y[i] = pd.read_csv(f"tmp/{store_name}_{part}_y_{task}.csv")
            if i != 0:
                stack_filenames[i - 1] = pd.read_csv(
                    f"tmp/{store_name}_{part}_filename.csv"
                )
        for i, val in tqdm(enumerate(X)):
            X[i] = X[i].rename(columns={"0": "features"})
            X[i] = X[i]["features"].str.split(",", expand=True).iloc[:, 2:].values

        print(f"X: {X[0].shape} | \t y: {y[0].shape}")
        return X, y, stack_filenames[0], stack_filenames[1]
