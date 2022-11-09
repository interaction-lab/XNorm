from email.mime import audio
import os
import wave
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import sys

# from pytube import YouTube

import torch
import torch.utils.data as data
from torchvision import transforms

import torchaudio
from transformers import HubertForCTC, Wav2Vec2Processor

import videotransforms

action_list = ['put-down','pick-up','open','rinse','take','close','turn-on','move','turn-off','get']

label_dict = {
	"absent":0,
	"being_helped_by_parents":1,
	"drinking_eating":2,
	"engaged_with_the_system":3,
	"looking_away":4,
	"not_in_seat_playing_in_background":5,
	"playing_with_toys":6,
	"talking_to_the_child's_self":7,
	"talking_to_their_parents":8,
	"touching_face":9
}

bundle = torchaudio.pipelines.HUBERT_BASE


def sample_frames(start_frame, stop_frame, num_frames=8):
	frame_list = range(start_frame, stop_frame+1)
	return [frame_list[i] for i in np.linspace(0, len(frame_list)-1, num_frames).astype('int')]


def load_rgb_frames(rgb_root, start_frame, stop_frame, num_frames):
	frames = []
	frame_list = sample_frames(start_frame, stop_frame, num_frames)
	for frame_idx in frame_list:
		frame_name = 'frame_'+str(frame_idx).zfill(10)+'.jpg'
		if 's5' in rgb_root:
			print("load rgb frames:", rgb_root, frame_name)
		abs_path = os.path.dirname(os.path.abspath(__file__))
		# print(abs_path)
		if cv2.imread(os.path.join(abs_path, rgb_root, frame_name)) is None:
			print("Truely read nothing!")
			print(rgb_root, frame_name)
			continue
		img = cv2.imread(os.path.join(abs_path, rgb_root, frame_name))[:, :, [2, 1, 0]]
		w,h,c = img.shape
		if w < 226 or h < 226:
			d = 226.-min(w,h)
			sc = 1+d/min(w,h)
			img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
		img = (img / 255.)*2 - 1
		frames.append(img)

	# print(np.asarray(frames, dtype=np.float32).ndim, "num dims for frames")
	return np.asarray(frames, dtype=np.float32)


def load_flow_frames(flow_u_root, flow_v_root, start_frame, stop_frame, num_frames):
	frames = []
	frame_list = sample_frames(start_frame, stop_frame, num_frames)
	for frame_idx in frame_list:
		frame_name = 'frame_'+str(frame_idx).zfill(10)+'.jpg'
		img_u = cv2.imread(os.path.join(flow_u_root, frame_name), cv2.IMREAD_GRAYSCALE)
		img_v = cv2.imread(os.path.join(flow_v_root, frame_name), cv2.IMREAD_GRAYSCALE)
		w,h = img_u.shape
		if w < 224 or h < 224:
			d = 224.-min(w,h)
			sc = 1+d/min(w,h)
			img_u = cv2.resize(img_u,dsize=(0,0),fx=sc,fy=sc)
			img_v = cv2.resize(img_v,dsize=(0,0),fx=sc,fy=sc)

		img_u = (img_u / 255.) * 2 - 1
		img_v = (img_v / 255.) * 2 - 1
		img = np.asarray([img_u, img_v]).transpose([1,2,0])
		frames.append(img)

	return np.asarray(frames, dtype=np.float32)

def load_audio(audio_root):
	# device = torch.device("cuda:0")
	waveform, sample_rate = torchaudio.load(audio_root)
	waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
	# average two channels
	# audio file has 2 channels, which represent stereo sound
	# average can convert both stereo and mono to a single channel
	averagedWaveform = torch.mean(waveform,0)
	# print("averagedWaveform: ", averagedWaveform)
	return averagedWaveform


def video_to_tensor(pic):
	"""Convert a ``numpy.ndarray`` to tensor.
	Converts a numpy.ndarray (T x H x W x C)
	to a torch.FloatTensor of shape (C x T x H x W)
	
	Args:
		 pic (numpy.ndarray): Video to be converted to tensor.
	Returns:
		 Tensor: Converted video.
	"""
	return torch.from_numpy(pic.transpose([3,0,1,2]))


class EPIC_Kitchens(data.Dataset):
	def __init__(self, csv_path, data_root, train, num_frames):
		self.dataset = pd.read_csv(csv_path)
		self.data_root = data_root
		self.num_frames = num_frames
		# bundle = torchaudio.pipelines.HUBERT_BASE
		# need audio path
		# self.audio
		self.audio = data_root

		if train:
			self.transform = transforms.Compose([
				videotransforms.RandomCrop(224),
				videotransforms.RandomHorizontalFlip()])
		else:
			self.transform = transforms.Compose([videotransforms.CenterCrop(224)])

	def __getitem__(self, index):
		participant_id = self.dataset['participant_id'][index]
		label = self.dataset['label'][index]
		video_name = self.dataset['video_name'][index]
		start_frame = int(self.dataset['start_frame'][index])
		stop_frame = int(self.dataset['stop_frame'][index])

		rgb_root = os.path.join(self.data_root, participant_id, label, video_name)
		print("rgb_root:", rgb_root)
		rgb_frames = load_rgb_frames(rgb_root, start_frame, stop_frame, self.num_frames)
		# if len(rgb_frames.ndim) < 4:
		# 	print("dim of frame in getitem:", rgb_root, rgb_frames.ndim)
		try:
			rgb_frames = video_to_tensor(self.transform(rgb_frames))
		except ValueError:
			print("ValueError", ValueError)
			print("rgb root: ", rgb_root, rgb_frames)
			sys.exit()

		# load audio
		audio_wav_root = os.path.join(self.data_root, participant_id, label, video_name, "{}.wav".format(video_name))
		audio_waveform_sample_rate = load_audio(audio_wav_root)

		label_index = label_dict[label]

		# now only return the rgb_frames and the audio_waveform
		return rgb_frames, audio_waveform_sample_rate, label_index


	def __len__(self):
		return len(self.dataset)