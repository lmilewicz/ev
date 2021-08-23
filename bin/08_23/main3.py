import torch
import torch.nn as nn

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
        x = self.gap(self.model(x))

        x = x.view(x.size(0), -1)

        return self.linear(x), None
    
    def phase_active(gene):
        """
        Determine if a phase is active.
        :param gene: list, gene describing a phase.
        :return: bool, true if active.
        """
        # The residual bit is not relevant in if a phase is active, so we ignore it, i.e. gene[:-1].
        return sum([sum(t) for t in gene[:-1]]) != 0


def demo():
    # genome = [[[1], [0, 1], [0, 0, 1], [0]]]

    # channels = [(3, 128)]

    # out_features = 10
    # data = torch.randn(16, 3, 32, 32)
    net = EvoNetwork()
    output = net(torch.autograd.Variable(data))

    print(output)


if __name__ == "__main__":
    demo()