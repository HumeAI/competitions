"""
Copyright (c) 2022 Marco Jiralerspong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

(MIT License (https://opensource.org/licenses/MIT) )
"""

import pickle
import argparse

from utils import get_activation_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_1', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    mu1, sigma1 = get_activation_statistics(args.samples_1, args.batch_size)
    dict = {
        "mu": mu1,
        "sigma": sigma1
    }

    try:
        file_path = f"fid/{args.name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(dict, f)

        print(f"Successfully saved activation stats in {file_path}")
    except Exception as e:
        print(e)
