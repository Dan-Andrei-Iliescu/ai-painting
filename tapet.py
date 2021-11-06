import torch
import os

import numpy as np

from skimage import io, color
from tqdm import tqdm
from fire import Fire


def main(
        shape_img="shape.png", app_img="app.png", res_img="res.png",
        num_patches=1024, num_trials=8):

    # Torch device (only CPU atm)
    device = torch.device("cpu")

    # I/O dirs
    data_dir = "data"
    result_dir = "results"

    # Paths
    shape_path = os.path.join(data_dir, shape_img)
    app_path = os.path.join(data_dir, app_img)
    result_path = os.path.join(result_dir, res_img)

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

    y_height = y.shape[1]
    y_width = y.shape[2]
    y = torch.from_numpy(y).to(device).float()

    # Normalize
    y[0] /= 100.
    y[1] = y[1] / 220. + 0.5
    y[2] = y[2] / 220. + 0.5

    res = torch.zeros_like(x) \
        + torch.Tensor([0.8, 0.5, 0.5]).to(device).unsqueeze(-1).unsqueeze(-1)

    # For decreasing kernel sizes
    num_scales = 8
    for scale_idx in tqdm(range(num_scales)):
        # Compute patch dimensions
        kernel_size = np.int16(2**(num_scales - scale_idx - 1))
        patch_size = 2*kernel_size+1

        # For selected patches at this size
        x_rows = torch.randint(x_height-patch_size,
                               (num_patches,)) + kernel_size
        x_cols = torch.randint(
            x_width-patch_size, (num_patches,)) + kernel_size

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

            x_patch = x[:,
                        row_idx-kernel_size:row_idx+kernel_size+1,
                        col_idx-kernel_size:col_idx+kernel_size+1]
            res_patch = res[:,
                            row_idx-kernel_size:row_idx+kernel_size+1,
                            col_idx-kernel_size:col_idx+kernel_size+1]

            # For a while, select random patches from y
            for _ in range(num_trials):
                y_row = torch.randint(
                    kernel_size, y_height-kernel_size-1, (1,))
                y_col = torch.randint(kernel_size, y_width-kernel_size-1, (1,))
                y_patch = y[:,
                            y_row-kernel_size:y_row+kernel_size+1,
                            y_col-kernel_size:y_col+kernel_size+1]

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

        print(f"\n{count}\n")

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


if __name__ == "__main__":
    Fire(main)
