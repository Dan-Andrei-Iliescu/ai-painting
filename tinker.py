import numpy as np

from PIL import Image, ImageCms
import torch
from torch import nn
from tqdm import tqdm

import sys



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


device = torch.device("cuda")

paint_path = "data/tenerife.png"
tex_path = "data/martyna_norm.png"
result_path = "results/tinker.png"

# Process inputs
srgb_profile = ImageCms.createProfile("sRGB")
lab_profile  = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(
	srgb_profile, lab_profile, "RGB", "LAB"
)
lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(
	lab_profile, srgb_profile, "LAB", "RGB"
)

x = Image.open(paint_path).convert("RGB")
x = ImageCms.applyTransform(x, rgb2lab_transform)
x = np.array(x)/255.
print(x.shape)
x_height = x.shape[0]
x_width = x.shape[1]
x_channels = x.shape[2]
x = torch.from_numpy(x).to(device)

y = Image.open(tex_path).convert("RGB")
y = ImageCms.applyTransform(y, rgb2lab_transform)
y = np.array(y)/255.
print(y.shape)
y_height = y.shape[0]
y_width = y.shape[1]
y_channels = y.shape[2]
y = torch.from_numpy(y).to(device)


num_patches = 4096
chosen_patches = 512
res = torch.zeros_like(x)


"""
# Locations in images
x_row_locs = torch.arange(x_height)
x_col_locs = torch.arange(x_width)
x_locs = torch.cartesian_prod(x_row_locs, x_col_locs)
x_num_locs = x_locs.shape[0]

y_row_locs = torch.arange(y_height)
y_col_locs = torch.arange(y_width)
y_locs = torch.cartesian_prod(y_row_locs, y_col_locs)
y_num_locs = y_locs.shape[1]


# For decreasing kernel sizes
num_kernels = 2
for kernel_idx in tqdm(range(num_kernels)):
	kernel_size = 2**(num_kernels - kernel_idx)
	patch_size = 2*kernel_size+1
	kernel = gaussian_kernel(kernel_size).to(device).view(
		patch_size, patch_size, 3)

	# Update probabilities
	x_diff = torch.zeros_like(x) + x
	x_prob = torch.zeros((x_num_locs,))

	order = torch.argsort(x_diff, descending=True)
	x_locs = x_locs[order[torch.randint(chosen_patches, (num_patches,))]]

	for x_loc in x_locs:
		# Check a patch is possible at location
		cond = (kernel_size <= x_loc[0] < x_height-kernel_size) \
			&& (kernel_size <= x_loc[1] < x_width-kernel_size)  
		if cond:



sys.exit(0)
"""

num_trials = 256

# For decreasing kernel sizes
num_kernels = 7
for kernel_idx in tqdm(range(num_kernels)):
	kernel_size = 2**(num_kernels - kernel_idx)
	patch_size = 2*kernel_size+1
	kernel = gaussian_kernel(kernel_size).to(device).view(
		patch_size, patch_size, 3)

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
		patches[patch_idx] = x[row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1]

	# Measure uniqueness
	for patch_idx in range(num_patches):
		test = torch.mean(torch.abs(patches[patch_idx]-patches), dim=(1, 2, 3))
		uniq[patch_idx] = torch.mean(torch.topk(test, k=2, largest=False)[0])

	# Choose unique ones
	order = torch.argsort(uniq, descending=True)
	perm = order[torch.randint(chosen_patches, (chosen_patches,))]
	x_rows = x_rows[perm]
	# 	+ torch.randint(-kernel_size, kernel_size, (chosen_patches,))
	x_cols = x_cols[perm]
	# 	+ torch.randint(-kernel_size, kernel_size, (chosen_patches,))

	count = 0
	for patch_idx in range(chosen_patches):
		row_idx = x_rows[patch_idx]
		col_idx = x_cols[patch_idx]
		x_patch = x[row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1]
		res_patch = res[row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1]
		# uniq[patch_idx] += torch.mean(torch.abs(x_patch - patches)**2)

		# Uniqueness of this patch
		
		"""
		res[row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1
		] += 0.02

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
			if new_diff < old_diff:
				res[row_idx-kernel_size:row_idx+kernel_size+1,
					col_idx-kernel_size:col_idx+kernel_size+1
				] = y_patch
				count += 1

				break

	print(f"\n{count}\n")


# Save output
img = res.cpu().detach().numpy()
img = Image.fromarray(np.uint8(img*255), mode="LAB")
#img = Image.fromarray(np.uint8(img*255))
img = ImageCms.applyTransform(img, lab2rgb_transform)
img.save(result_path)