import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LambdaLR # LR Scheduler
import training
import testing

def fit_model(net, epochs, device, trainloader, testloader, classes):
	# Loss Function & Optimizer and Step learning rate
	loss_fun = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
	scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

	EPOCHS = epochs
	# Starting Model Training
	for epoch in range(EPOCHS):
		print("EPOCH:", epoch+1)
		train_result = training.train(net, device, trainloader, optimizer, epoch, loss_fun)
		net = train_result['model']
		scheduler.step()

		val_result = testing.test(net, device, testloader, classes, loss_fun)

	# Training Finished
	result = {'model': net, 'val_acc': val_result['val_acc'], 'val_loss': val_result['val_loss'], 'train_acc': train_result['train_acc'], 'train_loss': train_result['train_loss']}
	print('Finished Training')
	return result