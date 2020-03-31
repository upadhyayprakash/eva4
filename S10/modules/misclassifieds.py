import matplotlib.pyplot as plt
import numpy as np


def get_mis_classified_objects(model, device, data_loader, no_of_images=25):
  data_loader_iterator = iter(data_loader)
  fail_count = 0
  failed_samples = []
  while fail_count <= no_of_images:
    data, target = data_loader_iterator.next()
    data, target = data.to(device), target.to(device)

    output = model(data)

    pred = output.argmax(dim=1, keepdim=True)
    failed_index = ~pred.eq(target.view_as(pred)).squeeze()

    failed_data = data[failed_index]
    failed_target = target[failed_index]
    failed_prediction = pred[failed_index]
    
    batch_fail_count = failed_data.size(dim=0)
    fail_count += batch_fail_count

    for count in range(batch_fail_count):
      failed_sample = {
          'data': failed_data[count].view(3, 32, 32).cpu().numpy().swapaxes(0, 2).swapaxes(0, 1),
          'target': failed_target[count],
          'prediction': failed_prediction[count].item()
      }

      failed_samples.append(failed_sample)

  return failed_samples

# Plot the Mis-classified Objects
def plot_mis_classified_objects(model, device, classes, data_loader):
  figure = plt.figure(figsize=(10, 10))
  num_of_images = 25
  failed_samples = get_mis_classified_objects(model, device, data_loader, num_of_images)
  for index in range(1, num_of_images + 1):
    ax = plt.subplot(5, 5, index)
    plt.tight_layout(pad=1.0)
    plt.axis('off')
    image = failed_samples[index - 1]['data']
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    plt.imshow(image)
    ax.set_title(f"target: {classes[failed_samples[index - 1]['target']]} \n prediction: {classes[failed_samples[index - 1]['prediction']]}")

  return failed_samples