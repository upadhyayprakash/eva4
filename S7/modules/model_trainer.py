import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LambdaLR # LR Scheduler
import training

def train_model(net, epochs, device, trainloader, loss_fun):
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

	EPOCHS = epochs
	# Starting Model Training
	for epoch in range(EPOCHS):
	    print("EPOCH:", epoch+1)
	    net = training.train(net, device, trainloader, optimizer, epoch, loss_fun)
	    scheduler.step()
	    # test(net, device, testloader)

	# Training Finished
	print('Finished Training')
	return net