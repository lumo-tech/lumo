from lumo import Saver
import time
import shutil


def test_save_load():
    save_root = './temp_saver'
    saver = Saver(save_root)
    epoch = 20
    max_keep = 5
    for i in range(epoch):
        time.sleep(0.01)
        saver.save_model(i, {'step': i}, {'meta_step': i}, is_best=(i == 5))
        if i % 2 == 0:
            saver.save_keypoint(i, {'step': i}, {'meta_step': i})
        else:
            saver.save_checkpoint(i, {'step': i}, {'meta_step': i}, max_keep=max_keep, is_best=(i % 5) == 0)

    assert len(saver.list_models()) == (epoch)

    state = saver.load_model(best_if_exist=True, with_meta=True)
    assert state[0]['step'] == 5 and state[1]['meta_step'] == 5

    state = saver.load_model(2)
    assert state['step'] == 2

    assert len(saver.list_keypoints()) == (epoch // 2)
    state = saver.load_checkpoint(best_if_exist=True)
    assert state['step'] == 15

    state = saver.load_checkpoint(0)  # max keep=5 ,11 , 13, 15, 17, 19 for it.
    assert state['step'] == 11

    assert len(saver.list_checkpoints()) == max_keep
    state = saver.load_keypoint(2)
    assert state['step'] == 4

    shutil.rmtree(save_root)
