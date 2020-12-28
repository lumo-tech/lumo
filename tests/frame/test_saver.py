"""

"""

from thexp.frame.saver import Saver

saver = Saver("./sav")

def test_max_to_keep():
    for i in range(5):
        saver.save_checkpoint(i,dict(a=i),dict(b=i))
    assert len(saver.find_checkpoints()) == saver.max_to_keep
    saver.clear_checkpoints()
    assert len(saver.find_checkpoints()) == 0

def test_keypoint():
    saver.clear_keypoints()
    for i in range(5):
        saver.save_keypoint(i,dict(a=i),dict(b=i))
    for i in range(5):
        assert saver.load_keypoint(i)["a"] == i
        assert saver.load_keypoint_info(i)["b"] == i

    assert len(saver.find_keypoints()) == 5
    saver.clear_keypoints()
    assert len(saver.find_keypoints()) == 0
