import internal.dataLoader.volumetricData as dataLoader
import internal.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

# Define the training loop
def train(model, train_loader, criterion, optimizer, num_epochs, scheduler,trainset_len,batch_size=64):
    # Set the model to training mode
    model.train()
    for epoch in range(num_epochs):
        with tqdm(total=trainset_len, desc='Epoch: ' + str(epoch+1) + "/" + str(num_epochs), unit='block') as prog_bar:
            for i, data in enumerate(train_loader):
                # Get inputs and labels
                inputs = data['input'].cuda()
                labels = data['label'].cuda()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print statistics
                prog_bar.set_postfix(**{'loss': loss.data.cpu().detach().numpy()})
                prog_bar.update(batch_size)
            scheduler.step()

def modelTest(model, test_loader, criterion):
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        running_loss = 0.0

        for i, data in enumerate(test_loader):
            # Get the inputs
            inputs = data['input'].cuda()
            labels = data['label'].cuda()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Update the running loss
            running_loss += loss.item()

        # Print the average loss for the test set
        test_loss = running_loss / len(test_loader)
        print('Test Loss: {:.8f}'.format(test_loss))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "volumes/porsche/porsche.bvp"
    save_path = 'trained/'
    # Define the dataset and dataloader
    batch_size = 1
    dataset = dataLoader.VolumetricData(path,0.8,64)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False)

    #trainset = dataLoader.TestingData(dataset.get_test_set())
    #test_loader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False)

    # Define the model, loss function, and optimizer
    model = models.Unet(4).cuda()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # Train the model
    num_epochs = 20
    train(model, train_loader, criterion, optimizer, num_epochs, scheduler,len(dataset),batch_size)
    modelTest(model , train_loader,criterion)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path + 'model3.pth')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/




