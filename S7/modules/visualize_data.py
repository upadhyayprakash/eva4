import matplotlib.pyplot as plt
import numpy as np

def visualize_batch(dataLoader, classes):
	dataiter = iter(dataLoader)
	images, labels = dataiter.next()
	plt.rcParams["figure.figsize"] = (6,6)

	print('Batch Grid')

	labelsList = labels.tolist()
	for index in range(0, images.size()[0]):
	    plt.subplot(4, 4, index+1)
	    # plt.axis('off')
	    img = images[index]
	    img = img / 2 + 0.5     # unnormalize
	    npimg = img.numpy()
	    plt.tight_layout(pad=1.0)
	    plt.imshow(np.transpose(npimg, (1,2,0)))
	    plt.text(1, -3, classes[labelsList[index]], fontsize=15)
