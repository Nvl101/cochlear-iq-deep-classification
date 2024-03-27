'''
run supervised training for image quality classification
- read configurations
- initiate the dataloaders
- initiate the classification network
- run training, output loss functions
'''
import torch
import config
from dataio.dataloader import create_dataloader
from networks.resnet_classifier import resnet18_classifier

# reading from configurations
dicoms = config.tracking_table['dicom_path'].to_list()
labels = config.tracking_table['label'].astype('int16').to_list()

# initiate the dataloaders
dataloader_dict = create_dataloader(
    dicoms, labels,
    config.dicom_dir,
    config.validation_size, config.test_size
)
training_dataloader, validation_dataloader, test_dataloader = dataloader_dict.get('dataloaders')
# some settings, move to config in future
learning_rate = 0.001
n_epoches = 3
# the classifier
model = resnet18_classifier
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    # training iteration
    for epoch in range(n_epoches):
        print(f'epoch # {epoch}')
        for images, labels in training_dataloader:
            labels_pred = model(images)
            loss = loss_function(labels_pred, labels)
            print(f'loss: {str(loss)}')
            # backpropagation
            optimizer.zero_grad()
            loss_function.backward()
            optimizer.step()
    print('-'*30)
    # validation
    partial_accuracies = []
    for images, labels in validation_dataloader:
        labels_pred = model(images)
        labels_raw = torch.argmax(labels, 1)
        labels_pred_raw = torch.argmax(labels_pred, 1)
        label_pred_correct = (labels_pred_raw == labels_raw).tolist()
        partial_accuracy = sum(label_pred_correct) / len(label_pred_correct)
        partial_accuracies.append(partial_accuracy)
    accuracy = sum(partial_accuracies) / len(partial_accuracies)
    print(f'accuracy: {accuracy}')
