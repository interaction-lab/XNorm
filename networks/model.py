from email.mime import audio
from typing import OrderedDict
import torch
import torch.nn as nn
from networks.pytorch_i3d import InceptionI3d

# move hubert_feature.py file code to here
import torchaudio
import torch
from transformers import HubertForCTC, Wav2Vec2Processor

# TODO: currently not able to import hubert_feature.py and have the code pasted below
# from hubert_feature import refactorWaveform

# start for hubert_feature.py
def refactorWaveform(audio_waveform_sample_rate):
	# print("audio_waveform_sample_rate",audio_waveform_sample_rate.shape)
	'''
	Args:
		audio_waveform_sample_rate: waveform
	Returns:
		1d array for audio feature from the waveform
	'''
	bundle = torchaudio.pipelines.HUBERT_BASE
	model = bundle.get_model()
	model = model.to(torch.device("cuda:0"))
	waveform = audio_waveform_sample_rate
	features, _ = model.extract_features(waveform)
	feature_array = features[-1].squeeze()
	feature_array_1d = torch.mean(feature_array,1)
	print("mean shape: ", feature_array_1d.shape)
	return feature_array_1d
# end for hubert_feature.py


class Classifier(nn.Module):
	def __init__(self, in_size, hidden_size, dropout, num_classes):
		super().__init__()

		self.dropout = nn.Dropout(dropout)
		self.fc1 = nn.Linear(in_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		x = self.dropout(self.fc1(x))
		x = torch.tanh(x)
		x = self.dropout(self.fc2(x))
		x = torch.tanh(x)
		x = self.dropout(self.fc3(x))

		return x


class EarlyFusion(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.rgb_enc = InceptionI3d(400, in_channels=3, dropout_rate=config.dropout)
		self.rgb_enc.load_state_dict(torch.load('checkpoints/rgb_imagenet.pt'))
		self.rgb_enc.replace_logits(config.hidden_size)

		# replace the classifier size with hidden_size + 768 (hubert feature size)
		self.out = Classifier(in_size=config.hidden_size + 768, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

	def forward(self, rgb_frames, audio_waveform_sample_rate):
		B = rgb_frames.size(0)
		rgb_features = self.rgb_enc(rgb_frames).view(B, -1)
		# get audio features from waveform
		audio_features = refactorWaveform(audio_waveform_sample_rate)

		return self.out(torch.cat([rgb_features, audio_features], dim=1))


class LateFusion(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.weight = config.weight

		self.rgb_enc = InceptionI3d(400, in_channels=3, dropout_rate=config.dropout)
		self.rgb_enc.load_state_dict(torch.load('checkpoints/rgb_imagenet.pt'))
		self.rgb_enc.replace_logits(config.hidden_size)
		self.rgb_out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

		self.flow_enc = InceptionI3d(400, in_channels=2, dropout_rate=config.dropout)
		self.flow_enc.load_state_dict(torch.load('checkpoints/flow_imagenet.pt'))
		self.flow_enc.replace_logits(config.hidden_size)
		self.flow_out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

	def forward(self, rgb_frames, flow_frames):
		B = rgb_frames.size(0)
		rgb_features = self.rgb_enc(rgb_frames).view(B, -1)
		flow_features = self.flow_enc(flow_frames).view(B, -1)

		return self.rgb_out(rgb_features)*self.weight+self.flow_out(flow_features)*(1-self.weight)


class RGB(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.rgb_enc = InceptionI3d(400, in_channels=3, dropout_rate=config.dropout)
		self.rgb_enc.load_state_dict(torch.load('checkpoints/rgb_imagenet.pt'))
		self.rgb_enc.replace_logits(config.hidden_size)
		self.rgb_out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

	def forward(self, rgb_frames, flow_frames):
		B = rgb_frames.size(0)
		rgb_features = self.rgb_enc(rgb_frames).view(B, -1)

		return self.rgb_out(rgb_features)


class Flow(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.flow_enc = InceptionI3d(400, in_channels=2, dropout_rate=config.dropout)
		self.flow_enc.load_state_dict(torch.load('checkpoints/flow_imagenet.pt'))
		self.flow_enc.replace_logits(config.hidden_size)
		self.flow_out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

	def forward(self, rgb_frames, flow_frames):
		B = rgb_frames.size(0)
		flow_features = self.flow_enc(flow_frames).view(B, -1)

		return self.flow_out(flow_features)


class XNorm(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.weight = config.weight
		self.ablation = config.ablation
		self.exchangelayers = ["Mixed_3b", "Mixed_4d", "Mixed_5c"]

		self.rgb_enc = InceptionI3d(400, in_channels=3, dropout_rate=config.dropout)
		self.rgb_enc.load_state_dict(torch.load('checkpoints/rgb_imagenet.pt'))
		self.rgb_enc.replace_logits(config.hidden_size)
		self.rgb_out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

		# define audio here - feature extraction? Where to do this?
		self.flow_enc = InceptionI3d(400, in_channels=2, dropout_rate=config.dropout)
		self.flow_enc.load_state_dict(torch.load('checkpoints/flow_imagenet.pt'))
		self.flow_enc.replace_logits(config.hidden_size)
		self.flow_out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

		self.rgb2flow_l1 = nn.Conv3d(256, 512, 1)
		self.rgb2flow_l2 = nn.Conv3d(512, 1024, 1)
		self.rgb2flow_l3 = nn.Conv3d(1024, 2048, 1)
		self.flow2rgb_l1 = nn.Conv3d(256, 512, 1)
		self.flow2rgb_l2 = nn.Conv3d(512, 1024, 1)
		self.flow2rgb_l3 = nn.Conv3d(1024, 2048, 1)

	def forward(self, rgb_frames, flow_frames): # B C T H W
		B = rgb_frames.size(0)
		x1, x2 = rgb_frames, flow_frames
		for endpoint in self.rgb_enc.VALID_ENDPOINTS:
			if endpoint in self.rgb_enc.end_points:
				layer1 = self.rgb_enc._modules[endpoint]
				layer2 = self.flow_enc._modules[endpoint]
				x1, x2 = layer1(x1), layer2(x2)

				if endpoint in self.exchangelayers:
					a1, b1 = torch.mean(x1, [1,2,3,4], True), torch.std(x1, [1,2,3,4], keepdim=True).add(1e-8)
					a1 = a1.repeat(1, x1.size(1), x1.size(2), x1.size(3), x1.size(4))
					b1 = b1.repeat(1, x1.size(1), x1.size(2), x1.size(3), x1.size(4))
					if endpoint == "Mixed_3b":
						a2, b2 = torch.chunk(self.rgb2flow_l1(x2), 2, dim=1)
					elif endpoint == "Mixed_4d":
						a2, b2 = torch.chunk(self.rgb2flow_l2(x2), 2, dim=1)
					else:
						a2, b2 = torch.chunk(self.rgb2flow_l3(x2), 2, dim=1)

					x1 = x1+((x1-a1)/b1)*b2+a2

					a2, b2 = torch.mean(x2, [1,2,3,4], True), torch.std(x2, [1,2,3,4], keepdim=True).add(1e-8)
					a2 = a2.repeat(1, x2.size(1), x2.size(2), x2.size(3), x2.size(4))
					b2 = b2.repeat(1, x2.size(1), x2.size(2), x2.size(3), x2.size(4))
					if endpoint == "Mixed_3b":
						a1, b1 = torch.chunk(self.flow2rgb_l1(x1), 2, dim=1)
					elif endpoint == "Mixed_4d":
						a1, b1 = torch.chunk(self.flow2rgb_l2(x1), 2, dim=1)
					else:
						a1, b1 = torch.chunk(self.flow2rgb_l3(x1), 2, dim=1)

					x2 = x2+((x2-a2)/b2)*b1+a1

		x1 = self.rgb_enc.logits(self.rgb_enc.dropout(self.rgb_enc.avg_pool(x1)))
		if self.rgb_enc._spatial_squeeze:
			x1 = x1.squeeze(3).squeeze(3)

		x2 = self.flow_enc.logits(self.flow_enc.dropout(self.flow_enc.avg_pool(x2)))
		if self.flow_enc._spatial_squeeze:
			x2 = x2.squeeze(3).squeeze(3)

		rgb_features = x1.view(B, -1)
		flow_features = x2.view(B, -1)

		return self.rgb_out(rgb_features)*self.weight+self.flow_out(flow_features)*(1-self.weight)


class GB(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.rgb_enc = InceptionI3d(400, in_channels=3, dropout_rate=config.dropout)
		self.rgb_enc.load_state_dict(torch.load('checkpoints/rgb_imagenet.pt'))
		self.rgb_enc.replace_logits(config.hidden_size)
		self.rgb_out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

		self.flow_enc = InceptionI3d(400, in_channels=2, dropout_rate=config.dropout)
		self.flow_enc.load_state_dict(torch.load('checkpoints/flow_imagenet.pt'))
		self.flow_enc.replace_logits(config.hidden_size)
		self.flow_out = Classifier(in_size=config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

		self.out = Classifier(in_size=2*config.hidden_size, hidden_size=config.hidden_size, dropout=config.dropout, num_classes=config.num_classes)

	def forward(self, rgb_frames, flow_frames):
		B = rgb_frames.size(0)
		rgb_features = self.rgb_enc(rgb_frames).view(B, -1)
		flow_features = self.flow_enc(flow_frames).view(B, -1)

		return self.rgb_out(rgb_features),self.flow_out(flow_features),self.out(torch.cat([rgb_features, flow_features], dim=1))