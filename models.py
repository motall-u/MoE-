import torch
import torch.nn as nn
import torch.optim as optim



# Define the expert model
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.softmax(self.layer2(x), dim=1)
    


# Define the gating model
class Gating(nn.Module):
    def __init__(self, input_dim,
                 num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(256, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)
        x = torch.softmax(self.layer4(x), dim=1)
       
        return x

    

class MoE(nn.Module):
    def __init__(self, trained_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)

        # Freezing the experts to ensure that they are not
        # learning when MoE is training.
        # Ideally, one can free them before sending the
        # experts to the MoE; in that case the following three
        # lines can be commented out.
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        input_dim = trained_experts[0].layer1.in_features
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x)

        # Calculate the expert outputs
        outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2)

        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1).expand_as(outputs)
        # print(weights.shape)
        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        # print("output shape"  , outputs.shape)
        # print("weights shape"  , weights.shape)
        return torch.sum(outputs * weights, dim=2)


if __name__== "__main__": 
    # Define the expert models
    experts = [Expert(input_dim=10, hidden_dim= 128, output_dim=2) for _ in range(10000)]

    # Define the MoE model
    moe = MoE(experts)
    print(moe)

    x = torch.randn(1 ,  10)
    print(moe(x))