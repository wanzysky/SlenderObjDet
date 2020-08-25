import io
import s3path
from PIL import Image

import detectron2.data.detection_utils as utils


def load_image_from_oss(path: s3path.S3Path, mode='rb', format=None):
    """

    Args:
        path:
        mode:
        format:

    Returns:

    """
    assert isinstance(path, s3path.S3Path)
    image = Image.open(io.BytesIO(path.open(mode=mode).read()))
    image = utils.convert_PIL_to_numpy(image, format)
    
    return image
