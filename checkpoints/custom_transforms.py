from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
import numpy as np
from torch import Tensor

@TRANSFORMS.register_module()
class LoadProposalsFromAnnotations(BaseTransform):
    def __init__(self):
        pass

    def transform(self, results: dict) -> dict:
        assert 'gt_bboxes' in results, 'results must contain key: gt_bboxes'
        # print('results[\'gt_bboxes\']:', results['gt_bboxes'])
        if (not isinstance(results['gt_bboxes'], (Tensor, np.ndarray))):
            # print('results[\'gt_bboxes\'].tensor:', results['gt_bboxes'].tensor)
            results['proposals'] = results['gt_bboxes'].tensor
        else:
            results['proposals'] = results['gt_bboxes']
        results['proposals_scores'] = np.full(len(results['gt_bboxes']), 0.99, dtype=np.float32)
        return results