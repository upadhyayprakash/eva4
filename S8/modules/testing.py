import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import progress_bar

# Evaluation module
test_losses = []
test_acc = []
predicted_labels_idx = []
result = dict()

def test(model, device, test_loader, classes, loss_fun):
  model.eval()
  correct = 0
  total = 0
  test_loss = 0
  
  dataiter = iter(test_loader)
  inputs, targets = dataiter.next()
  
  # Visualizing the Ground Truth
  # visualize_ground_truth(inputs, targets, classes)

  with torch.no_grad():
      for idx, (data, labels) in enumerate(test_loader, 0):
          # images, labels = data
          images, labels = data.to(device), labels.to(device)
          outputs = model(images)
          loss = loss_fun(outputs, labels)
          test_loss += loss.item()

          _, predicted = torch.max(outputs.data, 1)
          
          # if idx == 1:
          #   print('Predicted')
          #   print(predicted)
          
          total += labels.size(0)
          correct += predicted.eq(labels).sum().item()

          for pred in predicted:
            predicted_labels_idx.append(pred.item())
          
          progress_bar(idx, len(test_loader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                % (test_loss/(idx+1), 100.*correct/total, correct, total))

  # print('Accuracy of the network on the 10000 test images: %d %%' % (
  #     100 * correct / total))
  test_acc.append(100. * correct / len(test_loader.dataset))
  test_loss /= len(test_loader.dataset)
  test_losses.append(100. *test_loss)
  
  # creating response object
  # result['predicted_labels_idx'] = predicted_labels_idx
  result['val_acc'] = test_acc
  result['val_loss'] = test_losses
  
  return result


def visualize_ground_truth(images, labels, classes):
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
