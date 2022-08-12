import os
import sys
import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.misa import MISA
from networks.misa_function import DiffLoss, MSE, CMD

class AR_MISA_solver(nn.Module):
	def __init__(self, config):
		super(AR_MISA_solver, self).__init__()
		self.config = config

		# Initiate the networks
		self.model = MISA(config)

		# Setup the optimizers and loss function
		opt_params = list(self.model.parameters())
		self.optimizer = torch.optim.AdamW(opt_params, lr=config.learning_rate, weight_decay=config.weight_decay)
		self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=config.when, factor=0.5, verbose=False)
		self.criterion = nn.CrossEntropyLoss(reduction="mean")
		self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
		self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
		self.loss_diff = DiffLoss()
		self.loss_recon = MSE()
		self.loss_cmd = CMD()

		# Select the best ckpt
		self.best_val_metric = 0.

	def get_cmd_loss(self):
		loss = self.loss_cmd(self.model.x_1_shared, self.model.x_2_shared, 5)

		return loss

	def get_diff_loss(self):
		shared_1 = self.model.x_1_shared
		shared_2 = self.model.x_2_shared
		private_1 = self.model.x_1_private
		private_2 = self.model.x_2_private

		# Between private and shared
		loss = self.loss_diff(private_1, shared_1)
		loss += self.loss_diff(private_2, shared_2)

		# Across privates
		loss += self.loss_diff(private_1, private_2)

		return loss
	
	def get_recon_loss(self):
		loss = self.loss_recon(self.model.x_1_recon, self.model.x_1_orig)
		loss += self.loss_recon(self.model.x_2_recon, self.model.x_2_orig)

		return loss/2.0

	def update(self, rgb_frames, flow_frames, labels):
		self.train()
		self.optimizer.zero_grad()

		rgb_frames, flow_frames, labels = rgb_frames.cuda(), flow_frames.cuda(), labels.cuda()
		pred = self.model(rgb_frames, flow_frames)

		cls_loss = self.criterion(pred, labels)
		diff_loss = self.get_diff_loss()
		recon_loss = self.get_recon_loss()
		similarity_loss = self.get_cmd_loss()

		loss = cls_loss + \
			self.config.diff_weight * diff_loss + \
			self.config.recon_weight * recon_loss + \
			self.config.sim_weight * similarity_loss

		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

		self.optimizer.step()

	def val(self, val_loader):
		val_loss, val_acc = self.test(val_loader)
		self.save_best_ckpt(val_acc)
		self.scheduler.step(val_loss)

		return val_loss, val_acc

	def test(self, test_loader):
		with torch.no_grad():
			self.eval()
			preds, gt = [], []
			total_loss, total_samples = 0.0, 0
			for (rgb_frames, flow_frames, labels) in test_loader:
				rgb_frames, flow_frames, labels = rgb_frames.cuda(), flow_frames.cuda(), labels.cuda()
				pred = self.model(rgb_frames, flow_frames)
				loss = self.criterion(pred, labels)
				_, pred = torch.max(pred, 1)
				preds.append(pred)
				gt.append(labels)

				total_loss += loss.item()*labels.size(0)
				total_samples += labels.size(0)

			preds, gt = torch.cat(preds).cpu(), torch.cat(gt).cpu()
			acc = accuracy_score(np.array(gt), np.array(preds))
			loss = total_loss / total_samples

			self.print_metric([loss, acc])

			return loss, acc

	def load_best_ckpt(self):
		ckpt_name = os.path.join(self.config.ckpt_path, self.config.fusion + ".pt")
		state_dict = torch.load(ckpt_name)

		self.model.load_state_dict(state_dict['model'])

	def save_best_ckpt(self, val_metric):
		def update_metric(val_metric):
			if val_metric > self.best_val_metric:
				self.best_val_metric = val_metric
				return True
			return False

		if update_metric(val_metric):
			ckpt_name = os.path.join(self.config.ckpt_path, self.config.fusion +'.pt')
			torch.save({'model': self.model.state_dict()}, ckpt_name)

	def print_metric(self, metric):
		print('Loss: %.4f Acc: %.3f'%(metric[0], metric[1]))

	def run(self, train_loader, val_loader, test_loader):
		best_val_loss = 1e8
		patience = self.config.patience
		for epochs in range(1, self.config.num_epochs+1):
			print('Epoch: %d/%d' % (epochs, self.config.num_epochs))
			for _, (rgb_frames, flow_frames, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
				self.update(rgb_frames, flow_frames, labels)

			# Validate model
			val_loss, val_acc = self.val(val_loader)

			if val_loss < best_val_loss:
				patience = self.config.patience
				best_val_loss = val_loss
			else:
				patience -= 1
				if patience == 0:
					break

		# Test model
		self.load_best_ckpt()
		self.test(test_loader)
