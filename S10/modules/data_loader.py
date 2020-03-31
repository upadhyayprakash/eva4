import os
import torch
import torchvision
import data_transform as dt
import data_transform_albumentation as dta
from data_transform_albumentation import CustomAlbumentation

# cwd = os.getcwd()
# print('Current Working Directory')
# print(cwd)

cuda = torch.cuda.is_available() # returns True/False
print('GPU Available?', cuda)


def getTrainLoader(dataSource):
	trainset = None
	dataPath = './data/'+dataSource+'/'
	if dataSource == 'MNIST':
			trainset = torchvision.datasets.MNIST(root=dataPath, train=True, download=True, transform=dta.aug)
			# trainset = torchvision.datasets.MNIST(root=dataPath, train=True, download=True, transform=CustomAlbumentation())
			# trainset = torchvision.datasets.MNIST(root=dataPath, train=True, download=True, transform=dt.train_transform)

	if dataSource == 'CIFAR10':
			trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=True, transform=dta.aug)
			# trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=True, transform=CustomAlbumentation())
			# trainset = torchvision.datasets.CIFAR10(root=dataPath, train=True, download=True, transform=dt.train_transform)
	

	dataLoaderArguments = dict(shuffle=True, batch_size=128, num_workers=6, pin_memory=True) if cuda else dict(shuffle=True, batch_size=4)	
	
	trainloader = torch.utils.data.DataLoader(trainset, **dataLoaderArguments)
	return trainloader

def getTestLoader(dataSource):
	testset = None
	dataPath = './data/'+dataSource+'/'
	if dataSource == 'MNIST':
			# testset = torchvision.datasets.MNIST(root=dataPath, train=False, download=True, transform=dt.test_transform)
			testset = torchvision.datasets.MNIST(root=dataPath, train=False, download=True, transform=dta.basic_aug_transform)

	if dataSource == 'CIFAR10':
			# testset = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=True, transform=dt.test_transform)
			testset = torchvision.datasets.CIFAR10(root=dataPath, train=False, download=True, transform=dta.basic_aug_transform)
	
	dataLoaderArguments = dict(shuffle=True, batch_size=128, num_workers=6, pin_memory=True) if cuda else dict(shuffle=True, batch_size=4)	
	
	testloader = torch.utils.data.DataLoader(testset, **dataLoaderArguments)
	return testloader

def getClasses(dataSource):
	classes = []
	if dataSource == 'MNIST':
		classes = ('1','2','3','4','5','6','7','8','9','0')
	if dataSource == 'CIFAR10':
		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return classes