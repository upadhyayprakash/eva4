# Training module

train_losses = []
train_acc = []
result = dict()
def train(model, device, train_loader, optimizer, epoch, loss_fun):
  model.train()
  correct = 0
  processed = 0
  running_loss = 0.0
  for i, (data, labels) in enumerate(train_loader, 0):
    # get the inputs
    data, labels = data.to(device), labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(data)
    loss = loss_fun(outputs, labels) # CrossEntropyLoss
    train_losses.append(loss) # used during graph plot

    loss.backward()
    optimizer.step()

    # Get Values for Progress Statistics(Later)
    pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(labels.view_as(pred)).sum().item()
    processed += len(data)
    
    train_acc.append(100*correct/processed) # used during graph plot

    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

  return model