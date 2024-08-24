import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

from model_files.feature_fusion import get_yi, get_y_hat
from model_files.patch_extraction import get_patches

def get_patch_yi(model, image):
    """
    Calculates the feature representation of an image.
    :param model: The pre-trained CNN object
    :param image: The image
    :returns: The image's feature representation
    """
    transform = transforms.Compose([transforms.ToTensor()])

    y = []  # init Y

    patches = get_patches(image, stride=1024)

    for patch in patches:  # for every patch
        img_tensor = transform(patch)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor.double())
        yi = get_yi(model=model, patch=img_variable)
        y.append(yi)  # append Yi to Y

    y = np.vstack(tuple(y))

    y_hat = get_y_hat(y=y, operation="mean")  # create Y_hat with mean or max

    return y_hat