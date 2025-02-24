import os
import torch
import random
import numpy as np

from typing import (
    Dict,
    Mapping,
    Sequence,
)

def set_seeds(seed: int) -> int:
    torch.manual_seed(seed + 135)
    np.random.seed(seed + 235)
    random.seed(seed + 335)

    return seed + 435

def write_expert(observation, action, expert_output_path):
    directory = os.path.dirname(expert_output_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(expert_output_path, 'a') as file:
        observations_string = ' '.join(map(str, observation.flatten()))
        action_string = ' '.join(map(str, action.flatten()[-3:]))

        file.write(f'{observations_string} {action_string} \n')
