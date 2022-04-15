# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import json
import argparse

from pathlib import Path
from collections import defaultdict


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_file_path",
    type=str,
    default="./data_info.csv",
    help="Path to `data.info.csv` file.",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="./",
    help="Path to save the csv/json files for training.",
)


def create_splits(data_path, save_path):

    data_info = np.loadtxt(str(data_path), dtype=str, delimiter=",")

    split2files = defaultdict(list)
    for i, x in enumerate(data_info[1:]):
        filename = x[0][1:-1] + ".wav"
        split = x[1]
        gt = x[5:]
        if "Test" in x[1]:
            continue
        split2files[split].append(np.hstack([filename, gt]))

    # Writing csv files
    for split, data in split2files.items():
        np.savetxt(
            save_path / f"exvo_{split.lower()}.csv",
            np.array(data),
            delimiter=",",
            fmt="%s",
        )


#     # Run when speaker's id will become available on the test set
#     subject2files = defaultdict(list)
#     for i, x in enumerate(data_info[1:]):
#         split = x[1]
#         if 'Test' in split:
#             subject_id = x[2]
#             filename = x[0][1:-1] + '.wav'
#             gt = x[5:]
#             subject2files[subject_id].append(np.hstack([filename, gt]).tolist())

#     f = open('exvo_test_subject2files.json', 'w')
#     json.dump(subject2files, f)
#     f.close()


if __name__ == "__main__":
    args = parser.parse_args()
    create_splits(Path(args.data_file_path), Path(args.save_path))
