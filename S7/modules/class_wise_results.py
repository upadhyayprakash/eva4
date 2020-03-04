import torch

# Class-wise accuracy

def show_class_wise_results(model, testloader, device, classes):
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
	    print('Accuracy of %5s : %2d %%' % (
	        classes[i], 100 * class_correct[i] / class_total[i]))