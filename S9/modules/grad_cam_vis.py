import requests
from io import BytesIO
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
# for inline graph plotting

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

def visualize_grad_cam(model):
	cuda = torch.cuda.is_available() # returns True/False
	device = torch.device("cuda" if cuda else "cpu")


	url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png' # sample image for GradCam testing on Custom Model
	response = requests.get(url)
	pil_img = PIL.Image.open(BytesIO(response.content))
	pil_img


	# print('pre-processing the input')
	torch_img = transforms.Compose([
	    transforms.Resize((32, 32)),
	    transforms.ToTensor()
	])(pil_img).to(device)
	normed_torch_img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(torch_img)[None]


	# print('loading the models:')
	# resnet = models.resnet18(pretrained=True)
	# vgg = models.vgg16(pretrained=True)
	# print('loading the models Finished!')

	configs = [
	    dict(model_type='resnet', arch=model, layer_name='layer3'), # My Model
	    # dict(model_type='resnet', arch=resnet, layer_name='layer3'), # Reference 
	    # dict(model_type='vgg', arch=vgg, layer_name='features_29') # Reference
	]

	for config in configs:
	    config['arch'].to(device).eval()

	cams = [
	    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
	    for config in configs
	]

	images = []
	for gradcam, gradcam_pp in cams:
	    mask, _ = gradcam(normed_torch_img)
	    heatmap, result = visualize_cam(mask, torch_img)

	    mask_pp, _ = gradcam_pp(normed_torch_img)
	    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
	    
	    images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])

	images = make_grid(images, nrow=5, scale_each=False, padding=10, pad_value=140, range=5)
	output_dir = '/home/prakash/Prakash/EVA4/Session-9/Notebooks'
	os.makedirs(output_dir, exist_ok=True)
	output_name = 'output.jpeg'
	output_path = os.path.join(output_dir, output_name)
	# print(output_path)
	save_image(images, output_path)
	grad_cam_output = PIL.Image.open(output_path)
	print(grad_cam_output.size)
	resized_output = grad_cam_output.resize((500, 120))
	
	plt.imshow(np.asarray(resized_output))