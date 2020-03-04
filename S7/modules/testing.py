import torch
import matplotlib.pyplot as plt
import numpy as np

# Evaluation module
test_losses = []
test_acc = []
predicted_labels = []
result = dict()

def test(model, device, test_loader, classes):
  correct = 0
  total = 0
  
  dataiter = iter(test_loader)
  images, labels = dataiter.next()
  # Visualizing the Ground Truth
  visualize_ground_truth(images, labels, classes)
  images = images.to(device)
  outputs = model(images)

  with torch.no_grad():
      for data, labels in test_loader:
          # images, labels = data
          images, labels = data.to(device), labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          predicted_labels.append([classes[predicted[j]]
                              for j in range(16)])

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))

  result['predicted_labels'] = predicted_labels
  return result


def test1(model, device, test_loader, loss_fun, classes):
  dataiter = iter(test_loader)
  images, labels = dataiter.next()
  # Visualizing the Ground Truth
  visualize_ground_truth(images, labels, classes)
  images = images.to(device)
  outputs = model(images)

  model.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)

      test_loss += loss_fun(output, target)  # sum up batch loss
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  test_acc.append(100. * correct / len(test_loader.dataset))

  result['test_acc'] = test_acc
  result['test_losses'] = test_losses

  return result

def visualize_ground_truth(images, labels, classes):
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
