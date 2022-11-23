from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from pathlib import Path
from PIL import Image
import numpy as np


def Generate_ssim_mse(y_pred, RealPath):
    """
    given a dataset numpy array of the prediction image and the path to the Real Images,
    return two arrays containining the ssim and mse values whilst printing each values

    Parameters
    ----------
    y_pred : numpy.array
    RealPath : string

    Returns
    ----------
    Two arrays containing the SSIM and MSE values.

    Examples
    --------
    >>> import tools.metric as metrics
    >>> wrong_path = "a_wrong_path"
    >>> ssim_mse = metrics.Generate_ssim_mse([], wrong_path)
    >>> ssim_mse
    'Please select the right path'
    """
    try:
        Image.open(RealPath + "1.jpg")
    except BaseException:
        return "Please select the right path"
    #   return "Can't find the dataset, please set the directory again!"

    def scale2range(x, range):
        # Scale x into a range, both expected to be floats
        return (x - x.min()) * (max(range) - min(range)) / \
            (x.max() - x.min()) + min(range)

    # Load target and predicted images for comparison
    ssim_arr = []
    mse_arr = []
    for i in range(5):
        target_path = Path(f"{RealPath}{i + 1}.jpg")
        # predic_path = Path(f"storm_prediction{i + 1}.png")

        # Load images and cast to np arrays, defining datatypes as floats for scale2range function
        # target, predic = np.array(Image.open(target_path), dtype=np.float32), \
        # np.array(Image.open(predic_path), dtype=np.float32)
        target = np.array(Image.open(target_path), dtype=np.float32)
        predic = y_pred[i]

        # Scale predicted image to target range
        predic = scale2range(predic, [target.min(), target.max()])

        # Cast to uint8 to accommodate for consistent SSIM and MSE results
        target, predic = target.astype(np.uint8), predic.astype(np.uint8)

        # Compute metrics
        s, m = ssim(target, predic), mse(target, predic)
        ssim_arr.append(s)
        mse_arr.append(m)
        print(95 + i + 1, "th image", "ssim: ", s, "MSE", m)
    return ssim_arr, mse_arr


if __name__ == "__main__":
    import doctest
    doctest.testmod()
