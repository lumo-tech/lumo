from lumo import Trainer, TrainerParams
from lumo.exp.watch import Watcher
from lumo.proc.config import debug_mode


class MyTrainer(Trainer):
    pass


def trainer():
    params = TrainerParams()
    t = MyTrainer(params)
    return t


def test_exp():
    debug_mode()
    for i in range(10):
        t = trainer()
        t.train()
        print(t.exp.test_name)

    w = Watcher()
    df = w.load()
    print(df.columns)
    # print(sorted(list(df['test_name'])))
    assert len(df) == 10
