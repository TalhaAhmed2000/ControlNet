# This is an improved version and model of Holistically-Nested Edge Detection (HED) edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

# Note: For more detail regarding HED refer to https://arxiv.org/pdf/1504.06375


import os
import cv2
import torch
import numpy as np

# einops: Flexible and powerful tensor operations for readable and reliable code. Quote: Writing better code with PyTorch and einops 👌 Andrej Karpathy

from einops import rearrange
from annotator.util import annotator_ckpts_path


class DoubleConvBlock(torch.nn.Module):
    """
    DoubleConvBlock is a PyTorch module that applies a sequence of two or more convolutional layers with ReLU activation. It is used as a building block in the ControlNetHED_Apache2 model.
    """
    def __init__(self, input_channel, output_channel, layer_number):
        """
        Args:
        input_channel (int): Number of input channels.
        output_channel (int): Number of output channels.
        layer_number (int): Number of convolutional layers to apply.

        """
        
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        """
        Applies a sequence of two or more convolutional layers with ReLU activation, and returns the output tensor and a projection tensor.
        
        Args:
            x (torch.Tensor): The input tensor.
            down_sampling (bool, optional): If True, applies max pooling with a kernel size of (2, 2) and stride of (2, 2) to the input tensor before applying the convolutional layers. 
            Defaults to False.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor and a projection tensor.
        """
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):

    """
    ControlNetHED_Apache2 is a pytorch module for HED edge detection
    """

    def __init__(self):
        """
        The `__init__` method initializes the `ControlNetHED_Apache2` module. It sets up the necessary layers for the HED (Holistically-Nested Edge Detection) network, including:
        
        - `self.norm`: A learnable parameter that represents the normalization of the input tensor.
        - `self.block1`, `self.block2`, `self.block3`, `self.block4`, `self.block5`: DoubleConvBlock instances that apply a sequence of convolutional layers with ReLU activation. 
        These blocks have different input and output channel sizes, and different numbers of convolutional layers.
        
        The purpose of this initialization is to set up the network architecture for the HED model, which is used for edge detection in the `HEDdetector` class.
        """
     
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size = (1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        """
        Applies a sequence of convolutional blocks to the input tensor `x`, and returns the projection tensors from each block.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The projection tensors from the five convolutional blocks.
        """
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDdetector:
    """
    The `HEDdetector` class is a PyTorch module that performs Holistically-Nested Edge Detection (HED) on input images. It loads a pre-trained HED model from a remote location, 
    and provides a callable interface to apply the HED algorithm to input images.
    """

    def __init__(self):
        """
        The `__init__` method initializes the HED model by downloading the pre-trained weights if they are not already available locally, 
        and loading them into the `self.netNetwork` attribute.
        """
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
        modelpath = os.path.join(annotator_ckpts_path, "ControlNetHED.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        self.netNetwork = ControlNetHED_Apache2().float().cuda().eval()
        self.netNetwork.load_state_dict(torch.load(modelpath))

    def __call__(self, input_image):
        """
        Applies the Holistically-Nested Edge Detection (HED) algorithm to the input image and returns the resulting edge map. The `__call__` method takes an input image, preprocesses it, 
        passes it through the HED model, and returns the resulting edge map. The edge map is resized to match the input image dimensions and normalized to the range [0, 255].

        Args:
            input_image (numpy.ndarray): The input image as a numpy array with shape (H, W, C), where H is the height, W is the width, and C is the number of color channels.
        
        Returns:
            numpy.ndarray: The edge map as a numpy array with shape (H, W), where each pixel value represents the edge probability.
        """

        assert input_image.ndim == 3
        H, W, C = input_image.shape
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image.copy()).float().cuda()
            image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
            edges = self.netNetwork(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge


def nms(x, t, s):
    """
    Applies non-maximum suppression (NMS) to the input image `x` using the specified threshold `t` and Gaussian blur sigma `s`.
    
    The function first applies Gaussian blur to the input image `x` to smooth the edges. It then creates four different structuring elements (`f1`, `f2`, `f3`, `f4`) 
    and uses them to dilate the blurred image. The resulting dilated image is then compared to the original blurred image, and only the pixels where the dilated image is equal to 
    the original blurred image are kept. This effectively removes non-maximum edges.
    
    Finally, the function creates a new image `z` where pixels with values greater than the threshold `t` in the resulting image `y` are set to 255, and all other pixels are set to 0.
    
    Args:
        x (numpy.ndarray): The input image as a numpy array.
        t (float): The threshold value for non-maximum suppression.
        s (float): The standard deviation of the Gaussian blur.
    
    Returns:
        numpy.ndarray: The resulting image after non-maximum suppression.
    """
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z
