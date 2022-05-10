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
import argparse
from frechet_distance import calculate_frechet_distance
from utils import load_activation_statistics, get_activation_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_1', type=str)
    parser.add_argument('--samples_2', type=str)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    if ".pkl" in args.samples_1:
        mu1, sigma1 = load_activation_statistics(args.samples_1)
    else:
        mu1, sigma1 = get_activation_statistics(args.samples_1, args.batch_size)

    if ".pkl" in args.samples_2:
        mu2, sigma2 = load_activation_statistics(args.samples_2)
    else:
        mu2, sigma2 = get_activation_statistics(args.samples_2, args.batch_size)

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"FID between {args.samples_1} and {args.samples_2}: {fid:.2f}")