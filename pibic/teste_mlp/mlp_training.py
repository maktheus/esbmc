import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

def train():
    # XOR data
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Simple training loop
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')

    return model

def export_weights(model, filename):
    # Extract weights and biases
    w_hidden = model.hidden.weight.detach().numpy() # [4, 2]
    b_hidden = model.hidden.bias.detach().numpy()   # [4]
    w_out = model.output.weight.detach().numpy()    # [1, 4]
    b_out = model.output.bias.detach().numpy()      # [1]

    with open(filename, 'w') as f:
        f.write("#ifndef MLP_WEIGHTS_H\n")
        f.write("#define MLP_WEIGHTS_H\n\n")
        
        # Hidden weights: neuron by neuron
        f.write("float w_hidden[4][2] = {\n")
        for i in range(4):
            f.write(f"    {{{w_hidden[i][0]}f, {w_hidden[i][1]}f}},\n")
        f.write("};\n\n")
        
        # Hidden bias
        f.write("float b_hidden[4] = {\n")
        f.write(f"    {b_hidden[0]}f, {b_hidden[1]}f, {b_hidden[2]}f, {b_hidden[3]}f\n")
        f.write("};\n\n")
        
        # Output weights
        f.write("float w_out[4] = {\n")
        f.write(f"    {w_out[0][0]}f, {w_out[0][1]}f, {w_out[0][2]}f, {w_out[0][3]}f\n")
        f.write("};\n\n")
        
        # Output bias
        f.write(f"float b_out = {b_out[0]}f;\n\n")
        
        f.write("#endif\n")

if __name__ == "__main__":
    trained_model = train()
    export_weights(trained_model, "mlp_weights.h")
    print("Weights exported to mlp_weights.h")
