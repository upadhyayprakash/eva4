import os
import torch
import torchvision
import data_transform as dt

# cwd = os.getcwd()
# print('Current Working Directory')
# print(cwd)

cuda = torch.cuda.is_available() # returns True/False
print('GPU Available?', cuda)

def getTrainLoader(dataSource):
	trainset = None
	dataPath = './data/'+dataSource+'/'
	if dataSource == 'MNIST':
			trainset = torchvision.datasets.MNIST(root=dataPath, train=True, download=True, transform=dt.transform)

	if dataSource == 'CIFAR10':
			trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=True, transform=dt.transform)
	
	dataLoaderArguments = dict(shuffle=True, batch_size=16, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=4)	
	
	trainloader = torch.utils.data.DataLoader(trainset, **dataLoaderArguments)
	return trainloader

def getTestLoader(dataSource):
	testset = None
	dataPath = './data/'+dataSource+'/'
	if dataSource == 'MNIST':
			testset = torchvision.datasets.MNIST(root=dataPath, train=False, download=True, transform=dt.transform)

	if dataSource == 'CIFAR10':
			testset = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=True, transform=dt.transform)
	
	dataLoaderArguments = dict(shuffle=True, batch_size=16, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=4)	
	
	testloader = torch.utils.data.DataLoader(testset, **dataLoaderArguments)
	return testloader

def getClasses(dataSource):
	classes = []
	if dataSource == 'MNIST':
		classes = ('1','2','3','4','5','6','7','8','9','0')
	if dataSource == 'CIFAR10':
		classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return classes





# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)

# dataLoaderArguments = dict(shuffle=True, batch_size=16, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=4)

# trainloader = torch.utils.data.DataLoader(trainset, **dataLoaderArguments)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, **dataLoaderArguments)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')