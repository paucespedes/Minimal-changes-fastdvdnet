"""
Trains a FastDVDnet model.

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt

from models import FastDVDnet
from dataset import ValDataset
from dataloaders import train_dali_loader
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment
from train_common import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log, save_model_checkpoint
from PIL import Image

def main(**args):
	r"""Performs the main training loop
	"""

	# Load dataset
	print('> Loading datasets ...')
	dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
	loader_train = train_dali_loader(batch_size=args['batch_size'],\
									noisy_file_root=args['trainset_dir_noisy'],\
									original_file_root=args['trainset_dir_original'],\
									sequence_length=args['temp_patch_size'],\
									crop_size=args['patch_size'],\
									epoch_size=args['max_number_patches'],\
									random_shuffle=True,\
									temp_stride=3)

	num_minibatches = int(args['max_number_patches']//args['batch_size'])
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))

	# Init loggers
	writer, logger = init_logging(args)

	# Define GPU devices
	device_ids = [0]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = FastDVDnet()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()

	# Define loss
	criterion = nn.MSELoss(reduction='sum')
	criterion.cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer)

	# Training
	start_time = time.time()
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr
		print('\nlearning rate %f' % current_lr)

		# train

		for i, data in enumerate(loader_train, 0):

			# Pre-training step
			model.train()

			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer.zero_grad()

			# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
			# extract ground truth (central frame)
			img_train_original, img_train_noisy, gt_train_original, gt_train_noisy= normalize_augment(data[0]['data_original'], data[0]['data_noisy'], ctrl_fr_idx)
			N, _, H, W = img_train_original.size()

			# std dev of each sequence
			stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])
			# draw noise samples from std dev tensor
			noise = torch.zeros_like(img_train_original)
			noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

			# Add noise if 2 clean videos / leave img_train_noisy if noisy video in input
			imgn_train = img_train_noisy

			# Send tensors to GPU
			gt_train = gt_train_original.cuda(non_blocking=True)
			imgn_train = imgn_train.cuda(non_blocking=True)
			noise = noise.cuda(non_blocking=True)
			noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image

			# Evaluate model and optimize it
			out_train = model(imgn_train, noise_map)

			# Compute loss
			loss = criterion(gt_train, out_train) / (N*2)
			loss.backward()
			optimizer.step()

			save_training_steps = [16314, 16400, 16841, 17357, 17929, 20709, 20968]

			if training_params['step'] in save_training_steps:
				for x in range(16):
					for y in range(5):
						showImage2(img_train_original[x, 3*y: 3*y + 3, :, :], f"loss-{str(loss)}/Training-Step-{str(training_params['step'])}/Image-{str(x)}/Frame-{str(y)}/I")
						showImage2(imgn_train[x, 3*y: 3*y + 3, :, :], f"loss-{str(loss)}/Training-Step-{str(training_params['step'])}/Image-{str(x)}/Frame-{str(y)}/N")
						if ctrl_fr_idx == y:
							showImage2(out_train[x, :, :, :], f"loss-{str(loss)}/Training-Step-{str(training_params['step'])}/Image-{str(x)}/Frame-{str(y)}/O-(PSNR-{str(psnr(img_train_original[x, 3*y: 3*y + 3, :, :], out_train[x, : , :, :]))})")

			# if loss >= 16 and training_params['step'] >= 200:
			# 	showImage(gt_train[0], "Original " + str(0))
			# 	showImage(gt_train_noisy[0], "Noisy " + str(0))
			# 	showImage(out_train[0], "Out " + str(0))

			# for x in range(16):
			# 	# for y in range(5):
			# 	showImage(img_train_original[x, 3*ctrl_fr_idx: 3*ctrl_fr_idx + 3, :, :], "Original " + str(x) + ", frame " + str(ctrl_fr_idx+1) + "/5")
			# 	showImage(img_train_noisy[x, 3*ctrl_fr_idx: 3*ctrl_fr_idx + 3, :, :], "Noisy " + str(x) + ", frame " + str(ctrl_fr_idx+1) + "/5" + " PSNR: " + str(psnr(img_train_original[x, 3*ctrl_fr_idx: 3*ctrl_fr_idx + 3, :, :], img_train_noisy[x, 3*ctrl_fr_idx: 3*ctrl_fr_idx + 3, :, :])))

			# Results
			if training_params['step'] % args['save_every'] == 0:
				# Apply regularization by orthogonalizing filters
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)

				# Compute training PSNR
				log_train_psnr(out_train, \
								gt_train, \
								loss, \
								writer, \
								epoch, \
								i, \
								num_minibatches, \
								training_params)
			# update step counter
			training_params['step'] += 1

		# Call to model.eval() to correctly set the BN layers before inference
		model.eval()

		# Just in case validation fails
		save_model_checkpoint(model, args, optimizer, training_params, epoch)

		# Validation and log images
		validate_and_log(
						model_temp=model, \
						dataset_val=dataset_val, \
						valnoisestd=args['val_noiseL'], \
						temp_psz=args['temp_patch_size'], \
						writer=writer, \
						epoch=epoch, \
						lr=current_lr, \
						logger=logger, \
						trainimg=img_train_original
						)

		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model, args, optimizer, training_params, epoch)

	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)

def showImage(img, t):
	image = img.cpu()  # Extract the image at index i

	# Assuming the tensor is in [0, 1] range, you can convert it to [0, 255] range
	image = (image * 255).byte()

	# Convert tensor to numpy array and rearrange dimensions from [C, H, W] to [H, W, C]
	image = image.permute(1, 2, 0).numpy()

	# Display the image
	plt.imshow(image)
	plt.title(f"Image {t}")
	plt.show()

def showImage2(img, t):
	image = img.cpu()  # Extract the image at index i

	# Assuming the tensor is in [0, 1] range, you can convert it to [0, 255] range
	image = (image * 255).byte()

	# Convert tensor to numpy array and rearrange dimensions from [C, H, W] to [H, W, C]
	image = image.permute(1, 2, 0).numpy()

	# Display the image
	# plt.imshow(image)
	# plt.title(f"Image {t}")
	# plt.show()
	save_image_with_absolute_path(image, f"/home/pau/TFG/tests/data/new/{t}.png")

def save_image_with_absolute_path(image, absolute_path):
    try:
        # Save the image
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
        img_pil = Image.fromarray(image)
        img_pil.save(absolute_path)

        # print(f"Image saved successfully to {absolute_path}")

    except Exception as e:
        print(f"Error saving image: {e}")

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming values are normalized between 0 and 1
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_value.item()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters
	parser.add_argument("--batch_size", type=int, default=64, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=80, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=5,\
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=256000, \
						help="Maximum number of patches")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path of log files')
	parser.add_argument("--trainset_dir_original", type=str, default=None, \
					 help='path of trainset original videos')
	parser.add_argument("--trainset_dir_noisy", type=str, default=None, \
						help='path of trainset noisy videos')
	parser.add_argument("--valset_dir", type=str, default=None, \
						 help='path of validation set')
	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
