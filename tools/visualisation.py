import matplotlib.pyplot as plt

__all__ = ['show_batch']


def show_batch(dataset, nr=4, nc=4):
    """
    given a dataset with type dict with three keys including "Image ID", "image", "label"
    number of rows and columns, The function splits out images with that number of rows and columns

    Parameters
    ----------
    dataset : dict
    nr : float/int
    nc : float/int

    Returns
    ----------
    None

    Examples
    --------
    >>> image = {"Image ID": 1, "image": [[[1,2],[3,4],[5,6]]], "label": 1}
    >>> displaypic = show_batch(image)
    >>> displaypic
    """
    if not isinstance(dataset, type({})):
        print("dataset is not in correct format please insert a dictionary")
        return "dataset is not in correct format please insert a dictionary"
    fig, axarr = plt.subplots(nr, nc, figsize=(10, 10))
    for i in range(0, nr):
        for j in range(0, nc):
            sample = dataset['image'][0]
            try:
                axarr[i][j].imshow(sample)
            except BaseException:
                axarr[i][j].imshow(sample[0], cmap="gray")

    fig.tight_layout(pad=1.5)
    plt.show()
    return None


if __name__ == "__main__":
    import doctest
    doctest.testmod()
