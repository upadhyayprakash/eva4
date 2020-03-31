import torch
import matplotlib.pyplot as plt
import numpy as np

# Class-wise accuracy

def show_class_wise_results(model, testloader, device, classes):
	class_acc = []
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for data, labels in testloader:
			images, labels = data.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(4):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	for i in range(10):
		acc = 100 * class_correct[i] / class_total[i]
		class_acc.append(acc)
		print('Accuracy of %5s : %2d %%' % (
			classes[i], acc))

	visualize_classwise_accuracy(classes, class_acc)

def visualize_classwise_accuracy(classes, class_accuracy):
	plt.figure(figsize=(10, 6))
	num_x_point = np.arange(len(classes))
	plt.bar(num_x_point, class_accuracy, align='center', alpha=0.5, width=0.8, color='g')
	plt.xticks(num_x_point, classes)
	plt.ylabel('Accuracy')
	plt.title('Object Classes')

	# # Plot values of bars at their top
	# for i, v in enumerate(class_accuracy):
	# 	plt.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

	plt.show()