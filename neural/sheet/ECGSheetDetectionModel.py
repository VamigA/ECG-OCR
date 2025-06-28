import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ECGSheetDetectionModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features

		conv0 = self.backbone[0][0]
		self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
		with torch.no_grad():
			self.backbone[0][0].weight.copy_(conv0.weight.mean(1, keepdim=True))

		self.proj = nn.Linear(1280, 256)
		self.pos_embed = nn.Parameter(torch.zeros(1, 1225, 256))

		encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

		self.head = nn.Linear(1225 * 256, 8)

		nn.init.trunc_normal_(self.pos_embed, std=0.02)
		nn.init.xavier_uniform_(self.head.weight)
		nn.init.zeros_(self.head.bias)

	def forward(self, images):
		features = self.backbone(images)
		patch_embeddings = self.proj(features.flatten(2).transpose(1, 2))
		memory = self.encoder(patch_embeddings + self.pos_embed)

		out = self.head(memory.flatten(1))
		corners = torch.sigmoid(out) * 2 - 0.5
		return corners

__all__ = ('ECGSheetDetectionModel',)