from lr_finder import LRFinder # Reference package from: https://github.com/davidtvs/pytorch-lr-finder

# Using LRFinder
def lr_finder(net, optimizer, loss_fun, trainloader, testloader):
	# Using LRFinder
	lr_finder = LRFinder(net, optimizer, loss_fun, device='cuda')
	lr_finder.range_test(trainloader, val_loader=testloader, start_lr=1e-3, end_lr=0.1, num_iter=100, step_mode='exp')
	lr_finder.plot(log_lr=False)
	lr_finder.reset() # important to restore the model and optimizer's parameters to its initial state

	return lr_finder.history