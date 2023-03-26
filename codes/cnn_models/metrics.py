from .base import Metric
from .base import functional as F
import tensorflow as tf

SMOOTH = 1e-5

class PSNR(Metric):
    r"""Creates a criterion that measures the PSRN between the
    ground truth (gt) and the prediction (pr).

    Args:
		max val: the maximal pixel value in the image

    Returns:
        A callable ``psnr`` instance. Can be used in ``model.compile(...)`` function.

    Example:

    .. code:: python

        metric = PSNR()
        model.compile('SGD', loss=loss, metrics=[metric])
    """

    def __init__(
            self,
            max_val=None,
            name=None,
    ):
        name = name or 'psnr'
        super().__init__(name=name)
        self.max_val = max_val

    def __call__(self, gt, pr):
        return tf.image.psnr(gt, pr, max_val = self.max_val)

# aliases
psnr = PSNR()
