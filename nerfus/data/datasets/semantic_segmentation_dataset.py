"""
Semantic segmentation dataset.
"""

from typing import Dict
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfus.data.utils.data_utils import get_semantics_tensors_from_path


class SemanticSegmentationDataset(InputDataset):
    """Dataset that returns images and semantics.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["semantics"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
        self.semantics = self.metadata["semantics"]

    def get_metadata(self, data: Dict) -> Dict:
        # handle mask
        filepath = self.semantics.filenames[data["image_idx"]]
        semantic_label = get_semantics_tensors_from_path(filepath=filepath, scale_factor=self.scale_factor)

        return {"semantics": semantic_label}
