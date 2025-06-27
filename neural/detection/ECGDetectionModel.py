import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ECGDetectionModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features

		conv0 = self.backbone[0][0]
		self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
		with torch.no_grad():
			self.backbone[0][0].weight.copy_(conv0.weight.mean(1, keepdim=True))

		self.proj = nn.Linear(1280, 384)
		self.pos_embed = nn.Parameter(torch.zeros(1, 1225, 384))
		self.object_queries = nn.Parameter(torch.randn(13, 384))

		encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=8, batch_first=True)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
		decoder_layer = nn.TransformerDecoderLayer(d_model=384, nhead=8, batch_first=True)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

		self.bbox_head = nn.Linear(384, 4)
		self.presence_head = nn.Linear(384, 1)
		self.mmps_head = nn.Linear(384, 1)

		nn.init.trunc_normal_(self.pos_embed, std=0.02)
		nn.init.xavier_uniform_(self.bbox_head.weight)
		nn.init.zeros_(self.bbox_head.bias)
		nn.init.xavier_uniform_(self.presence_head.weight)
		nn.init.zeros_(self.presence_head.bias)

	def forward(self, images):
		batch_size = images.size(0)
		features = self.backbone(images)

		patch_embeddings = self.proj(features.flatten(2).transpose(1, 2))
		memory = self.encoder(patch_embeddings + self.pos_embed)
		queries = self.object_queries.unsqueeze(0).repeat(batch_size, 1, 1)
		decoder_out = self.decoder(queries, memory)

		bbox_output = torch.sigmoid(self.bbox_head(decoder_out[:, :12, :]))
		presence_output = self.presence_head(decoder_out[:, :12, :]).squeeze(-1)
		mmps_output = self.mmps_head(decoder_out[:, 12, :]).squeeze(-1)
		return (bbox_output, presence_output, mmps_output)

__all__ = ('ECGDetectionModel',)