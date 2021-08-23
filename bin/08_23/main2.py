#%% 
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

# import pyro
# import pyro.distributions as dist
# import pyro.poutine as poutine

from torchsummary import summary




class EvoNetwork(nn.Module):
    def __init__(self):
        """
        Network constructor.
        :param genome: depends on decoder scheme, for most this is a list.
        :param channels: list of desired channel tuples.
        :param out_features: number of output features.
        :param decoder: string, what kind of decoding scheme to use.
        """
        super().__init__()

        layers = []

        layers.append(nn.Linear(28 * 28 * 1, 10))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(10, 10))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(10, 10))


        self.model = nn.Sequential(*layers)

        # We accumulated some unwanted gradient information data with those forward passes.
        self.model.zero_grad()


    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = self.gap(self.model(x))
        x = self.model(x)
        return F.log_softmax(x)
    
    def phase_active(gene):
        """
        Determine if a phase is active.
        :param gene: list, gene describing a phase.
        :return: bool, true if active.
        """
        # The residual bit is not relevant in if a phase is active, so we ignore it, i.e. gene[:-1].
        return sum([sum(t) for t in gene[:-1]]) != 0

batch_size_train = 64
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=True, 
    transform=torchvision.transforms.ToTensor()), batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=False, 
    transform=torchvision.transforms.ToTensor()), batch_size=batch_size_test, shuffle=True)

# print(mnist_dataset)

# train_images, train_labels = mnist_dataset #, (test_images, test_labels)
# train_images, test_images = train_images / 255.0, test_images / 255.0

dtype = torch.float32


import time

learning_rate = 0.01
device = torch.device("cuda:0")
### Normal
model = EvoNetwork()
# output = model(torch.autograd.Variable(train_loader))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
log_interval = 490
n_epochs = 10
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(1, n_epochs + 1)]

start_time = time.time()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step() 
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, n_epochs + 1):
    train(epoch)
    # test()

print('Execution Time: %s' % (time.time()-start_time))
summary(model, (1, 28, 28))

# import matplotlib.pyplot as plt

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# fig

# model.compile(optimizer=nn.keras.optimizers.Adam(learning_rate), loss=nn.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
# model.fit(x=train_images,
#             y=train_labels,
#             epochs=10,
#             use_multiprocessing=True,
#             batch_size=64)

# model.summary()

### Pyro
# model_pyro = get_model((28, 28, 1), dtype=dtype, layerType=DenseFlipout)

# model_pyro.compile(optimizer=nn.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
# model_pyro.fit(x=train_images,
#             y=train_labels,
#             epochs=10,
#             use_multiprocessing=True,
#             batch_size=64)

# # model_pyro.summary()

# # %%
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)

# with torch.no_grad():
#     # model.to(torch.device("cpu"))
#     output = model(example_data)

# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Prediction: {}".format(
#         output.data.max(1, keepdim=True)[1][i].item()))
#     plt.xticks([])
#     plt.yticks([])
# fig
# # %%
