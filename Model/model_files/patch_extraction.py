from skimage.util import view_as_windows

def get_patches(image_mat, stride):
    """
    Extract patches rom an image
    :param image_mat: The image as a matrix
    :param stride: The stride of the patch extraction process
    :returns: The patches
    """
    window_shape = (128, 128, 3)
    windows = view_as_windows(image_mat, window_shape, step=stride)
    patches = []
    for m in range(windows.shape[0]):
        for n in range(windows.shape[1]):
            patches += [windows[m][n][0]]
    return patches