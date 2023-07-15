#!/usr/bin/env python

import argparse
import struct

import numpy as np
from sklearn.datasets import make_blobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-d', '--dim', type=int, default=100)
    parser.add_argument('-n', '--num_samples', type=int, default=1000000)
    parser.add_argument('-c', '--centers', type=int, default=10)
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)

    args = parser.parse_args()
    X, y = make_blobs(n_samples=args.num_samples,
                      n_features=args.dim,
                      random_state=args.seed,
                      centers=args.centers)

    random_generator = np.random.RandomState(seed=args.seed)
    centers = random_generator.rand(args.centers, args.dim)

    epsilon = args.epsilon

    with open('data.dat', 'wb') as data_file:
        data_file.write(struct.pack('3i', args.num_samples, args.dim, args.centers))
        data_file.write(struct.pack('d', epsilon))
        data_file.write(X.astype(np.float64).tobytes())
        data_file.write(centers.astype(np.float64).tobytes())
        data_file.write(y.astype(np.int32).tobytes())


if __name__ == "__main__":
    main()
