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

def visualize_grad_cam(model, image_list, classes):
	cuda = torch.cuda.is_available() # returns True/False
	device = torch.device("cuda" if cuda else "cpu")


	# url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png' # sample image for GradCam testing on Custom Model
	# response = requests.get(url)
	# pil_img = PIL.Image.open(BytesIO(response.content))
	# pil_img

	master_img_arr = []
	for idx in range(len(image_list)):

		pil_img = image_list[idx]['data']
		pil_img = (pil_img - pil_img.min()) / (pil_img.max() - pil_img.min())
		pil_img = (pil_img * 255).astype(np.uint8)
		pil_img = transforms.ToPILImage(mode='RGB')(pil_img)
		target_label = image_list[idx]['target']
		prediction_label = image_list[idx]['prediction']


		# print('pre-processing the input')
		torch_img = transforms.Compose([
			transforms.Resize((32, 32)),
			transforms.ToTensor()
		])(pil_img).to(device)
		normed_torch_img = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(torch_img)[None]


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

		for i in range(len(images)):
			master_img_arr.append({'image': images[i], 'target_label': target_label, 'prediction_label': prediction_label})

	
	# === Using Matplotlib
	figure = plt.figure(figsize=(10, 50))
	num_of_images = 125
	for index in range(1, num_of_images + 1):
		ax = plt.subplot(25, 5, index)
		plt.tight_layout(pad=1.0)
		plt.axis('off')
		image = master_img_arr[index - 1]['image']
		target_label = master_img_arr[index - 1]['target_label']
		prediction_label = master_img_arr[index - 1]['prediction_label']
		image = image.view(3, 32, 32).cpu().numpy().swapaxes(0, 2).swapaxes(0, 1)
		image = (image - image.min()) / (image.max() - image.min())
		image = (image * 255).astype(np.uint8)
		plt.imshow(image)
		ax.set_title(f"target: {classes[target_label]} \n prediction: {classes[prediction_label]}")

	# Storing Image as output
	output_dir = '/home/prakash/Prakash/EVA4/Session-10/Notebooks'
	os.makedirs(output_dir, exist_ok=True)
	output_name = 'output_misclassified_gradcam.jpeg'
	output_path = os.path.join(output_dir, output_name)
	figure.savefig(output_path, bbox_inches="tight")