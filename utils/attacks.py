import copy
import torch
import numpy as np

def sign_flipping_attack(weights, attack_value=-1):
    perturbed_weights = copy.deepcopy(weights)
    for k in perturbed_weights.keys():
        perturbed_weights[k] = perturbed_weights[k] * attack_value
    print(perturbed_weights)
    return perturbed_weights


def additive_noise_attack(weights, device, seed=42):
    perturbed_weights = copy.deepcopy(weights)
    for k in perturbed_weights.keys():
        noise = torch.from_numpy(np.random.default_rng(seed=seed).normal(loc=0.0, scale=1.0, size=perturbed_weights[k].shape )).to(device)
        perturbed_weights[k] = perturbed_weights[k] + noise
    print(perturbed_weights)
    return perturbed_weights


def same_value_attack(weights, attack_value=1):
    perturbed_weights = copy.deepcopy(weights)
    for k in perturbed_weights.keys():
        perturbed_weights[k] = torch.from_numpy(np.full(shape=perturbed_weights[k].shape, fill_value=attack_value))
    print(perturbed_weights)
    return perturbed_weights


class LabelFlip(object):
    """Flip labels of the dataset."""

    def __call__(self, label):
        
        # Flipping 7 and 5
        if label == 7:
            return 5
        if label == 5:
            return 7

        # Flipping 4 and 2
        if label == 4:
            return 2
        if label == 2:
            return 4

        return label
