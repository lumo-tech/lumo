
 - 调用 Trainer 的 load_state_dict 或是 load_model_state_dict 时，如果 model 重写了 "_load_state_dict"，那么会优先调用该函数，而不是 "model.load_state_dict" 函数。