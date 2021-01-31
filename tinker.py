import sys
import torch
import argparse

import numpy as np

from scipy.special import comb
from PIL import Image, ImageCms
from torch import nn
from tqdm import tqdm
from skimage import io, color


def gaussian_kernel(size, sigma=2., dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.
    
    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * np.sqrt(2 * np.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    kernel = kernel.view(1, *kernel.size())

    return kernel


def geom_e_max(p, n, trials=10):
	"""
		Expected maximum of <n> geometric i.i.d. r.v. with success prob <p>
	"""
	vals = np.zeros((trials,))
	for trial in range(trials):
		vals[trial] = np.amax(np.random.geometric(p, n))

	return np.mean(vals)


# Input args
parser = argparse.ArgumentParser(description='Thief')
parser.add_argument('--paint_img', type=str, default='mona_lisa.png')
parser.add_argument('--tex_img', type=str, default='martyna_norm.png')
parser.add_argument('--num_patches', type=int, default=512)
parser.add_argument('--pivot', type=float, default=0.43)
parser.add_argument('--num_trials', type=int, default=256)
args = parser.parse_args()

paint_path = f"data/{args.paint_img}"
tex_path = f"data/{args.tex_img}"
num_patches_static = args.num_patches
pivot = args.pivot
num_trials = args.num_trials

device = torch.device("cuda")
result_path = "results/tinker.png"


# Process images
x = io.imread(paint_path)[:, :, :3]
x = color.rgb2lab(x)
x = np.array(x)

x_height = x.shape[0]
x_width = x.shape[1]
x_channels = x.shape[2]
x = torch.from_numpy(x).to(device)

y = io.imread(tex_path)[:, :, :3]
y = color.rgb2lab(y)
y = np.array(y)

y_height = y.shape[0]
y_width = y.shape[1]
y_channels = y.shape[2]
y = torch.from_numpy(y).to(device)

res = torch.zeros_like(x) + torch.Tensor([50., 0., 0.]).to(device)


# For decreasing kernel sizes
num_kernels = 7
for kernel_idx in tqdm(range(num_kernels)):
	kernel_size = np.int16(2*2**(num_kernels - kernel_idx))
	patch_size = 2*kernel_size+1
	kernel = gaussian_kernel(kernel_size).to(device).view(
		patch_size, patch_size, 3)

	num_patches = np.uint16(num_patches_static * np.sqrt(2)**(kernel_idx))
	prob = pivot / (1.33**kernel_idx)
	e_max = geom_e_max(prob, num_patches, trials=100)
	print(kernel_size)
	print(num_patches)
	print(e_max)

	# For selected patches at this size
	x_rows = torch.randint(x_height-patch_size, (num_patches,)) + kernel_size
	x_cols = torch.randint(x_width-patch_size, (num_patches,)) + kernel_size
	uniq = torch.zeros_like(x_rows, dtype=torch.float32).to(device)

	# Get all the patches
	patches = torch.zeros([num_patches, patch_size, patch_size, x_channels],
		dtype=torch.float32).to(device)
	for patch_idx in range(num_patches):
		row_idx = x_rows[patch_idx]
		col_idx = x_cols[patch_idx]
		patch = x[row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1]

		# Normalize
		# patch -= torch.amin(patch)
		# patch /= torch.amax(patch)
		patches[patch_idx] = patch

	# Measure uniqueness
	for patch_idx in range(num_patches):
		# test = torch.mean(torch.abs(patches[patch_idx]-patches)**2, dim=(1, 2, 3))
		# uniq[patch_idx] = torch.mean(torch.topk(test, k=2, largest=False)[0])
		# uniq[patch_idx] = torch.mean(test)
		patch_mean = torch.mean(patches[patch_idx], dim=(0, 1))
		patch_var = torch.mean(torch.abs(patch-patch_mean)**2)
		uniq[patch_idx] = patch_var


	# Choose unique ones via geometric distribution
	order = torch.argsort(uniq, descending=True)
	perm = torch.zeros((num_patches,), dtype=torch.long).geometric_(p=prob)\
		.to(device) % num_patches
	x_rows = x_rows[order[perm]] \
	 	+ torch.randint(-kernel_size, kernel_size, (num_patches,))
	x_cols = x_cols[order[perm]] \
	 	+ torch.randint(-kernel_size, kernel_size, (num_patches,))

	
	count = 0
	for patch_idx in range(num_patches):
		row_idx = x_rows[patch_idx]
		col_idx = x_cols[patch_idx]

		row_idx = max(row_idx, kernel_size)
		col_idx = max(col_idx, kernel_size)
		row_idx = min(row_idx, x_height-kernel_size-1)
		col_idx = min(col_idx, x_width-kernel_size-1)

		x_patch = x[row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1]
		res_patch = res[row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1]

		"""
		# Uniqueness of this patch
		res[row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1
		] += torch.Tensor([0., 0.3, 0.]).to(device)
		"""
		
		# For a while, select random patches from y
		for idx in range(num_trials):
			y_row = torch.randint(kernel_size, y_height-kernel_size-1, (1,))
			y_col = torch.randint(kernel_size, y_width-kernel_size-1, (1,))
			y_patch = y[y_row-kernel_size:y_row+kernel_size+1,
				y_col-kernel_size:y_col+kernel_size+1]

			# Check if patch from y is a better fit to x
			old_diff = torch.mean(torch.abs(res_patch - x_patch))
			new_diff = torch.mean(torch.abs(y_patch - x_patch))
			if new_diff < old_diff or kernel_idx == 0:
				res[row_idx-kernel_size:row_idx+kernel_size+1,
					col_idx-kernel_size:col_idx+kernel_size+1
				] = y_patch
				count += 1

				break

	print(f"\n{count}\n")
	

# Save output
img = res.cpu().detach().numpy()
img = color.lab2rgb(img)
img = np.uint8(img*255)
io.imsave(fname=result_path, arr=img)