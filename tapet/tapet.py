import torch
import argparse

import numpy as np

from piqa import SSIM as IQA
from piqa.utils.functional import gaussian_kernel
from skimage import io, color
from tqdm import tqdm


# Transformation
class Net(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
		self.opt = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.1)

	def forward(self, x):
		x = self.conv(x)
		return x

	def train_step(self, loss):
		loss.backward()
		self.opt.step()


# Input args
parser = argparse.ArgumentParser(description='Thief')
parser.add_argument('--shape_img', type=str, default='river_1.png')
parser.add_argument('--app_img', type=str, default='munch_desp.png')
parser.add_argument('--num_patches', type=int, default=2048)
parser.add_argument('--num_trials', type=int, default=32)

args = parser.parse_args()

shape_path = f"data/{args.shape_img}"
app_path = f"data/{args.app_img}"
num_patches = args.num_patches
num_trials = args.num_trials

device = torch.device("cuda")
result_path = "results/tapet.png"
trans_path = "results/trans.png"

loss = IQA().to(device)


# Process images
x = io.imread(shape_path)[:, :, :3]
x = color.rgb2lab(x)
x = np.array(x)
x = np.transpose(x, axes=[2, 0, 1])

x_channels = x.shape[0]
x_height = x.shape[1]
x_width = x.shape[2]
x = torch.from_numpy(x).to(device).float()

# Normalize
x[0] /= 100.
x[1] = x[1] / 220. + 0.5
x[2] = x[2] / 220. + 0.5


y = io.imread(app_path)[:, :, :3]
y = color.rgb2lab(y)
y = np.array(y)
y = np.transpose(y, axes=[2, 0, 1])

y_channels = y.shape[0]
y_height = y.shape[1]
y_width = y.shape[2]
y = torch.from_numpy(y).to(device).float()

# Normalize
y[0] /= 100.
y[1] = y[1] / 220. + 0.5
y[2] = y[2] / 220. + 0.5


res = torch.zeros_like(x) \
	+ torch.Tensor([0.8, 0.7, 0.5]).to(device).unsqueeze(-1).unsqueeze(-1)

# Transformation
# net = Net().to(device)

# For decreasing kernel sizes
num_scales = 8
for scale_idx in tqdm(range(num_scales)):
	# Compute patch dimensions
	kernel_size = np.int16(2**(num_scales - scale_idx - 1))
	patch_size = 2*kernel_size+1

	# For selected patches at this size
	x_rows = torch.randint(x_height-patch_size, (num_patches,)) + kernel_size
	x_cols = torch.randint(x_width-patch_size, (num_patches,)) + kernel_size

	# Get all the patches in the shape image
	x_patches = torch.zeros([num_patches, x_channels, patch_size, patch_size],
		dtype=torch.float32).to(device)
	for patch_idx in range(num_patches):
		row_idx = x_rows[patch_idx]
		col_idx = x_cols[patch_idx]
		x_patches[patch_idx] = x[:,
			row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1]

	count = 0
	for patch_idx in range(num_patches):
		row_idx = x_rows[patch_idx]
		col_idx = x_cols[patch_idx]

		"""
		row_idx = max(row_idx, kernel_size)
		col_idx = max(col_idx, kernel_size)
		row_idx = min(row_idx, x_height-kernel_size-1)
		col_idx = min(col_idx, x_width-kernel_size-1)
		"""

		x_patch = x[:,
			row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1]
		res_patch = res[:,
			row_idx-kernel_size:row_idx+kernel_size+1,
			col_idx-kernel_size:col_idx+kernel_size+1]
		
		# For a while, select random patches from y
		for idx in range(num_trials):
			y_row = torch.randint(kernel_size, y_height-kernel_size-1, (1,))
			y_col = torch.randint(kernel_size, y_width-kernel_size-1, (1,))
			y_patch = y[:,
				y_row-kernel_size:y_row+kernel_size+1,
				y_col-kernel_size:y_col+kernel_size+1]

			"""
			# Check if patch from y is a better fit to x
			old_diff = 1. - loss(
				res_patch.unsqueeze(0), x_patch.unsqueeze(0))
			new_diff = 1. - loss(
				y_patch.unsqueeze(0), x_patch.unsqueeze(0))
			"""
			
			# Check if patch from y is a better fit to x
			old_diff = torch.mean(torch.abs(res_patch - x_patch))
			new_diff = torch.mean(torch.abs(y_patch - x_patch))
			

			# net.train_step(old_diff)
			# print(old_diff+new_diff)

			if new_diff < old_diff or scale_idx == 0:
				res[:,
					row_idx-kernel_size:row_idx+kernel_size+1,
					col_idx-kernel_size:col_idx+kernel_size+1
				] = y_patch
				count += 1

			"""
			loss = torch.mean(torch.abs(
				x.unsqueeze(0) - net(res.unsqueeze(0))))
			net.train_step(loss)
			"""

	print(f"\n{count}\n")

# trans_res = net(res.unsqueeze(0))[0]

# Unnormalize
res[0] *= 100.
res[1] = (res[1] - 0.5) * 220.
res[2] = (res[2] - 0.5) * 220.

# Save output
img = res.cpu().detach().numpy()
img = np.transpose(img, axes=[1, 2, 0])
img = color.lab2rgb(img)
img = np.uint8(img*255)
io.imsave(fname=result_path, arr=img)


"""
# Unnormalize
res = trans_res
res[0] *= 100.
res[1] = (res[1] - 0.5) * 220.
res[2] = (res[2] - 0.5) * 220.

# Save output
img = res.cpu().detach().numpy()
img = np.transpose(img, axes=[1, 2, 0])
img = color.lab2rgb(img)
img = np.uint8(img*255)
io.imsave(fname=trans_path, arr=img)
"""