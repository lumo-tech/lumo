"""

"""

import torch


class BatchEmbedding():
    """用于将多个 batch 的输入一同添加到 tensorboard 中"""
    def __init__(self, writer, global_step=None, tag="default"):
        self.writer = writer
        self.global_steps = global_step
        self.tag = tag
        self.mats = []
        self.metadatas = []
        self.label_imgs = []

    def add_embedding(self, mat, metadata=None, label_img=None):
        self.mats.append(mat.detach().cpu())
        self.metadatas.append(metadata.detach().cpu())
        self.label_imgs.append(label_img.detach().cpu())

    def flush(self, max_len=None):
        self.writer.add_embedding(torch.cat(self.mats[:max_len]),
                                  torch.cat(self.metadatas[:max_len]),
                                  self.label_imgs,
                                  tag=self.tag,
                                  global_step=self.global_steps)

