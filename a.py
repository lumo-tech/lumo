import time
from dataclasses import dataclass
import copy
import torch
from fairseq.modules.transformer_layer import TransformerEncoderLayer
from lightseq.training.ops.pytorch.transformer_encoder_layer import LSTransformerEncoderLayer
from contiguous_params import ContiguousParams


def get_time():
    '''CUDA同步并获取当前时间'''
    torch.cuda.synchronize(device="cuda:0")
    return time.time()


def ls_config_to_fs_args(config):
    '''将LightSeq的config转换为Fairseq的args'''

    @dataclass
    class Args:
        encoder_embed_dim: int
        encoder_ffn_embed_dim: int
        encoder_attention_heads: int
        dropout: float
        attention_dropout: float
        activation_dropout: float
        encoder_normalize_before: bool

    args = Args(config.hidden_size, config.intermediate_size, config.nhead, config.hidden_dropout_ratio,
                config.attn_prob_dropout_ratio, config.activation_dropout_ratio, config.pre_layer_norm)
    return args


def train(model, inputs, masks, contiguous=False):
    '''训练过程'''
    model.to(device="cuda:0")
    model.train()
    if contiguous:
        parameters = ContiguousParams(model.parameters())
        opt = torch.optim.Adam(parameters.contiguous(), lr=1e-3)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fw_time, bw_time, step_time = 0, 0, 0
    for epoch in range(1000):
        opt.zero_grad()
        start_time = get_time()
        outputs = model(inputs, masks)
        loss = torch.square(outputs).mean()
        fw_time += get_time() - start_time
        start_time = get_time()
        loss.backward()
        bw_time += get_time() - start_time
        start_time = get_time()
        opt.step()
        step_time += get_time() - start_time
    if epoch % 200 == 0:
        print("epoch {:>3d}: loss = {:>5.3f}".format(epoch, loss))
    return fw_time, bw_time, step_time


if __name__ == "__main__":
    # 定义LightSeq的config
    config = LSTransformerEncoderLayer.get_config(max_batch_tokens=4096, max_seq_len=256, hidden_size=128,
                                                  intermediate_size=512, nhead=16, attn_prob_dropout_ratio=0.1,
                                                  activation_dropout_ratio=0.1, hidden_dropout_ratio=0.1,
                                                  pre_layer_norm=True, fp16=False,
                                                  local_rank=0)
    # 将LightSeq的config转换为Fairseq的args
    args = ls_config_to_fs_args(config)
    # 随机生成输入
    bsz, sl = 50, 80
    inputs = torch.randn(bsz, sl, config.hidden_size).to(device="cuda:0")
    masks = torch.zeros(bsz, sl).to(device="cuda:0")
    # 定义LightSeq模型并训练
    ls_model = LSTransformerEncoderLayer(config)
    ls_fw_time, ls_bw_time, ls_step_time = train(ls_model, inputs, masks)
    # 定义连续化参数的LightSeq模型并训练
    config_cont = copy.deepcopy(config)
    ls_model_cont = LSTransformerEncoderLayer(config_cont)
    ls_c_fw_time, ls_c_bw_time, ls_c_step_time = train(ls_model_cont, inputs, masks, contiguous=True)
    inputs = inputs.transpose(0, 1)
    masks = masks > 0.5
    # 定义Fairseq模型并训练
    fs_model = TransformerEncoderLayer(args)
    fs_fw_time, fs_bw_time, fs_step_time = train(fs_model, inputs, masks)
    # 定义连续化参数的Fairseq模型并训练
    fs_model_cont = TransformerEncoderLayer(args)
    fs_c_fw_time, fs_c_bw_time, fs_c_step_time = train(fs_model_cont, inputs, masks, contiguous=True)
    print("LightSeq time:{:.3f}s, {:.3f}s, {:.3f}s".format(ls_fw_time, ls_bw_time, ls_step_time))
    print("LightSeq (cont) time:  {:.3f}s, {:.3f}s, {:.3f}s".format(ls_c_fw_time, ls_c_bw_time, ls_c_step_time))
    print("Fairseq time:          {:.3f}s, {:.3f}s, {:.3f}s".format(fs_fw_time, fs_bw_time, fs_step_time))
    print("Fairseq (cont) time:   {:.3f}s, {:.3f}s, {:.3f}s".format(fs_c_fw_time, fs_c_bw_time, fs_c_step_time))
