import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_sizes=(256, 128), num_classes=10, dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]  # [B, 1, 28, 28] -> [B, 784]
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        # Output raw logits; CrossEntropyLoss applies log-softmax internally
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLP()
    print(model)
    print("trainable params:", count_params(model))
    out = model(torch.randn(4, 1, 28, 28))
    print("output shape:", out.shape)  # expect [4, 10]