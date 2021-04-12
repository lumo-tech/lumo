from lumo import Saver

saver = Saver('./sav')
saver.save_checkpoint(1, {1, 2, 3})
saver.save_checkpoint(2, {1, 2, 3})
saver.save_checkpoint(3, {1, 2, 3})
saver.save_checkpoint(4, {1, 2, 3}, is_best=True)
saver.save_keypoint(3, {1, 2, 3})
saver.save_keypoint(3, {1, 2, 3})
fn = saver.save_model(4, {1, 2, 3}, is_best=True)

print(fn)

print(saver.load_state_dict(fn))

# saver.load_state_dict()
print(saver.list_models())
print(saver.list_checkpoints())
print(saver.list_keypoints())

