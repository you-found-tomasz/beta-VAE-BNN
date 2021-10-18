import torch
from vae_wrapper import single_inference
import numpy as np
from PIL import Image

im = Image.open("example/class_0/Circuit_52_Stimulus_0_frame_1.png")
imgArray = np.array(im).astype(np.float32)
imgArray[imgArray == 255] = 1
imgArray2 = torch.from_numpy(imgArray)
z = single_inference(imgArray2.unsqueeze(0))