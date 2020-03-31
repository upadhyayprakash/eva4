import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LambdaLR # LR Scheduler
import training
import testing
from lr_finder import LRFinder # Custom package

def fit_model(net, epochs, device, trainloader, testloader, classes, loss_fun, optimizer, scheduler):

	EPOCHS = epochs
	# Starting Model Training
	for epoch in range(EPOCHS):
		print("EPOCH:", epoch+1)
		train_result = training.train(net, device, trainloader, optimizer, epoch, loss_fun)
		
		net = train_result['model']
		
		val_result = testing.test(net, device, testloader, classes, loss_fun)
		scheduler.step(val_result['val_loss'][-1]/100) # should be called after Validation

	# Training Finished
	result = {'model': net, 'val_acc': val_result['val_acc'], 'val_loss': val_result['val_loss'], 'train_acc': train_result['train_acc'], 'train_loss': train_result['train_loss']}
	print('Finished Training')
	return result