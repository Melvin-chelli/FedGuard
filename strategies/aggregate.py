from functools import reduce
from typing import List, Tuple

import numpy as np
import torch
import copy

from flwr.common import NDArray, NDArrays
from utils.models import VAE


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def aggregate_median(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute median weight of each layer
    median_w: NDArrays = [
        np.median(np.asarray(layer), axis=0) for layer in zip(*weights)  # type: ignore
    ]
    return median_w



def aggregate_krum(
    results: List[Tuple[NDArrays, int]], to_keep: int
) -> NDArrays:
    """Choose one parameter vector according to the Krum fucntion.
    If to_keep is not None, then MultiKrum is applied.
    """
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    # For each client, take the n-2 closest parameters vectors
    num_closest = max(1, len(weights) - 2)
    closest_indices = []
    for i, _ in enumerate(distance_matrix):
        closest_indices.append(
            np.argsort(distance_matrix[i])[1 : num_closest + 1].tolist()  # noqa: E203
        )

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]  # noqa: E203
        best_results = [results[i] for i in best_indices]
        return aggregate(best_results)

    # Return the index of the client which minimizes the score (Krum)
    return weights[np.argmin(scores)]



def _compute_distances(weights: List[NDArrays]) -> NDArray:
    """Compute distances between vectors.
    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = np.array(
        [np.concatenate(p, axis=None).ravel() for p in weights]  # type: ignore
    )
    distance_matrix = np.zeros((len(weights), len(weights)))
    for i, _ in enumerate(flat_w):
        for j, _ in enumerate(flat_w):
            delta = flat_w[i] - flat_w[j]
            norm = np.linalg.norm(delta)  # type: ignore
            distance_matrix[i, j] = norm**2
    return distance_matrix


def flatten_weights(weights):
    weights = [np.array(layer).flatten() for layer in weights]
    flatten_weights = np.concatenate(weights, axis=0)
            
    return flatten_weights


def aggregate_spectral(results: List[Tuple[NDArrays, int]], vae, device) -> NDArrays:
    """Compute weighted average."""
    # Using the VAE to compute reconstruction score of local updates
    np.random.seed(1)
    vae.eval()

    weights = [weights for weights, num_examples in results]
    num_examples = [num_examples for weights, num_examples in results]

    user_one_d = []
    for user_weights in weights:
        flat_weights = flatten_weights(user_weights)
        surrogate_idx = np.random.choice(len(flat_weights), size=10000, replace=False)
        surrogate = np.array(flat_weights)[surrogate_idx]
        user_one_d.append(surrogate)

    torch.tensor(user_one_d).to(device)

    scores = vae.test(user_one_d, device=device)
    print("scores: ", scores)
    score_avg = np.mean(scores)
    print("score_avg: ", score_avg)

    benign_indices = []
    for i, score in enumerate(scores):
        if score <= score_avg:
            benign_indices.append(i)

    print("benign_indices: ", benign_indices)

    weights = [weights[i] for i in benign_indices]
    num_examples = [num_examples[i] for i in benign_indices]
    num_examples_total = sum([num_example for num_example in num_examples])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in zip(weights, num_examples)
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime