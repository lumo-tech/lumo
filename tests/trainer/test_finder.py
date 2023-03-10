import random

from lumo import Trainer, ParamsType, TrainerParams, Experiment
from lumo.exp import finder
from lumo.proc.config import debug_mode


class ATrainer(Trainer):

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)


class BTrainer(Trainer):

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)


def test_finder():
    debug_mode()

    for i in range(5):
        params = TrainerParams()
        params.epoch = i
        params.rnd = random.random()
        ATrainer(params).train()
        BTrainer(params).train()
    all_tests = finder.list_all()
    # print([ATrainer.generate_exp_name(), BTrainer.generate_exp_name()])
    print(ATrainer.__exp_name__)
    assert len(all_tests) == len({ATrainer.generate_exp_name(), BTrainer.generate_exp_name()})
    assert ATrainer.generate_exp_name() in all_tests
    assert BTrainer.generate_exp_name() in all_tests
    assert len(all_tests[ATrainer.generate_exp_name()]) == 5
    assert len(all_tests[BTrainer.generate_exp_name()]) == 5

    assert isinstance(all_tests[ATrainer.generate_exp_name()][0], Experiment)
    for exp in all_tests[ATrainer.generate_exp_name()]:
        params = TrainerParams().from_yaml(exp.properties['params.yaml'])
        assert params.hash() == exp.properties['params_hash']
        assert finder.find_path_from_test_name(exp.test_name) == exp.test_root
