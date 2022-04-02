from pprint import pprint

from lumo import Params

params = Params()  # A sub class of `omegaconf.DictConfig`
params.from_args()
params.from_dict(dict(epoch=10, batch_size=128))
params.from_args(['dataset=cifar10'])  # when empty, means parse from sys.argv
# params.from_hydra(config_path=..., config_name=...)  # A integration of hydra

# params.from_dict()
# params.from_json()
# params.from_yaml()

print(params['epoch'])
print(params.batch_size)
pprint(params.to_dict())

print(params.to_json())
print(params.to_yaml())
