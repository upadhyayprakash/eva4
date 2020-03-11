# Training module
import torch
from utils import progress_bar
from prefetcher import data_prefetcher

train_losses = []
train_acc = []
result = dict()
def train(model, device, train_loader, optimizer, epoch, loss_fun):
  model.train()
  correct = 0
  processed = 0
  train_loss = 0.0
  total = 0
  reg = 1e-6

  for i, (data, labels) in enumerate(train_loader, 0):
    # get the inputs
    # prefetcher = data_prefetcher(train_loader)
    # data, labels = prefetcher.next()

    data, labels = data.to(device), labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(data)
    loss = loss_fun(outputs, labels) # CrossEntropyLoss

    # Adding the Regularization L2 Loss
    l2_penalty = 0
    for param in model.parameters():
      l2_penalty += 0.5 * reg * torch.sum(torch.pow(param, 2))

    loss += l2_penalty


    train_loss += loss.item()

    loss.backward()
    optimizer.step()

    # Get Values for Progress Statistics(Later)
    pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(labels.view_as(pred)).sum().item()
    processed += len(data)
    
    total += labels.size(0)
    
    progress_bar(i, len(train_loader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
      % (train_loss/(i+1), 100.*correct/total, correct, total))


  train_acc.append(100*correct/processed) # used during graph plot
  train_loss /= len(train_loader.dataset)
  train_losses.append(100. *train_loss) # used during graph plot

  result['model'] = model
  result['train_acc'] = train_acc
  result['train_loss'] = train_losses

  return result