from enum import Enum
from typing import Callable, Optional, Union

import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent


class ExtendedFieldHeadNames(Enum):
    """Extended possible field outputs"""

    SEMANTIC_UNCERTAINTY = "semantic_uncertainty"
    EPISTEMIC_SEMANTIC_UNCERTAINTY = "epistemic_semantic_uncertainty"
    ALEATORIC_SEMANTIC_UNCERTAINTY = "aleatoric_semantic_uncertainty"

