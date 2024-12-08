import torch
from torch import nn
class BBOX_Network(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=64):
        super(BBOX_Network, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU()
        )

        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.residual_layer = nn.Linear(hidden_dim, hidden_dim)

        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, bboxes):
        # Get shape of objectness
        box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 6).to(bboxes.device)
        box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0] # width
        box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1] # height
        box_hw[:, :, 2] = box_hw[:, :, 0] * box_hw[:, :, 1]
        box_hw[:, :, 3] = box_hw[:, :, 0] / box_hw[:, :, 1]
        box_hw[:, :, 4] = box_hw[:, :, 0] + box_hw[:, :, 1]
        box_hw[:, :, 5] = torch.sqrt(box_hw[:, :, 0] ** 2 + box_hw[:, :, 1] ** 2)

        x = box_hw
        x = self.input_layer(x)
        hidden = self.hidden_layers(x)
        hidden = hidden + self.residual_layer(x)
        interacted_features = self.interaction_layer(hidden)
        output = self.output_layer(interacted_features)
        return output
