import matplotlib.pyplot as plt
import numpy as np

font = {'size'   : 12}
plt.rc('font', **font)

def visualize_batch(dataLoader, classes):
	dataiter = iter(dataLoader)
	images, labels = dataiter.next()
	plt.rcParams["figure.figsize"] = (6,6)

	print('Batch Grid')

	labelsList = labels.tolist()
	for index in range(0, 16):
	    plt.subplot(4, 4, index+1)
	    # plt.axis('off')
	    img = images[index]
	    img = img / 2 + 0.5     # unnormalize
	    npimg = img.numpy()
	    plt.tight_layout(pad=1.0)
	    plt.imshow(np.transpose(npimg, (1,2,0)))
	    plt.text(1, -3, classes[labelsList[index]], fontsize=15)

def visualize_train_test_results(train_acc, train_loss, val_acc, val_loss):
	# Visualizing Training and Testing Results

	fig, axs = plt.subplots(2,2,figsize=(15,10))
	axs[0, 0].plot(train_loss, color='red', linewidth=2)
	axs[0, 0].set_title("Training Loss")
	axs[1, 0].plot(train_acc, color='green', linewidth=2)
	axs[1, 0].set_title("Training Accuracy")
	axs[0, 1].plot(val_loss, color='red', linewidth=2)
	axs[0, 1].set_title("Test Loss")
	axs[1, 1].plot(val_acc, color='green', linewidth=2)
	axs[1, 1].set_title("Test Accuracy")