import numpy as np
from matplotlib import pyplot as plt
from lumo.contrib.itertools import window


def make_grid_from_files(files, pad=1, ncol=None, padvalue=0):
    return make_grid_from_nplist([plt.imread(f) for f in files], pad, ncol, padvalue)


def make_grid_from_nplist(vals, pad=1, ncol=None, padvalue=0):
    return make_grid_from_stacked_imgs(np.stack(vals), pad, ncol, padvalue)


def make_grid_from_stacked_imgs(imgs: np.ndarray, pad=1, ncol=None, padvalue=0):
    x = imgs.shape[0]

    ws = ncol
    if ws is None:
        ws = min(int(np.ceil(np.sqrt(x))), 8)
    else:
        ws = min(x, ws)
    hs = x // ws + int((x % ws) != 0)

    h, w = imgs.shape[1], imgs.shape[2]
    nh = hs * (h + pad) + pad
    nw = ws * (w + pad) + pad
    if len(imgs.shape) == 3:
        nimgs = np.zeros((nh, nw), dtype=imgs.dtype) + padvalue
    else:
        nimgs = np.zeros((nh, nw, imgs.shape[-1]), dtype=imgs.dtype) + padvalue

    for row, sub_imgs in enumerate(window(imgs, ws, ws)):
        for col, img in enumerate(sub_imgs):
            nimgs[pad + row * (h + pad):pad + row * (h + pad) + h,
            pad + col * (w + pad):pad + col * (w + pad) + w, ] = img
    return nimgs

vr = decord.VideoReader('v_HulaHoop_g07_c04.avi',width=32,height=32)
plt.figure(dpi=100)
plt.imshow(make_grid_from_nplist(vr[::4].asnumpy(),ncol=8))