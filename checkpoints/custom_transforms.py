from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
import numpy as np
from torch import Tensor

@TRANSFORMS.register_module()
class LoadProposalsFromAnnotations(BaseTransform):
    def __init__(self):
        pass

    def transform(self, results: dict) -> dict:
        # print("Enter LoadProposalsFromAnnotations.transform")
        # print("results:", results)
        assert 'gt_bboxes' in results, 'results must contain key: gt_bboxes'
        # print('results[\'gt_bboxes\']:', results['gt_bboxes'])
        if (not isinstance(results['gt_bboxes'], (Tensor, np.ndarray))):
            # print('results[\'gt_bboxes\'].tensor:', results['gt_bboxes'].tensor)
            results['proposals'] = results['gt_bboxes'].tensor
        else:
            results['proposals'] = results['gt_bboxes']

        # results['proposals'] = np.array([[22, 95, 22+256, 95+108]], dtype=np.float32)

        # resize proposals
        factor = np.array([
            results['scale_factor'][0], results['scale_factor'][1],
            results['scale_factor'][0], results['scale_factor'][1]
        ], dtype=np.float32)

        results['proposals'] = np.multiply(results['proposals'], factor)
        results['proposals_scores'] = np.full(len(results['proposals']), 0.99, dtype=np.float32)
        # print(results['proposals'])

        # results['proposals'] = np.array([[385.3159, 335.9352, 860.2880, 719.0645],], dtype=np.float32)
        # results['proposals_scores'] = np.full(1, 0.99, dtype=np.float32)
        return results