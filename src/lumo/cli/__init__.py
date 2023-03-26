import fire


def rerun(test_name, **kwarg):
    """
    rerun a test
        lumo rerun <test_name>
        lumo rerun <test_name> --device=0

    Args:
        test_name:

    Returns:

    """
    from lumo.exp.watch import Watcher
    from lumo import Experiment
    w = Watcher()
    df = w.load()
    df = df[df['test_name'] == test_name]
    if len(df) == 0:
        print(f'{test_name} not found')
        exit(1)
    else:
        exp = Experiment.from_cache(df.iloc[0].to_dict())
        exp.rerun([f'--{k}={v}' for k, v in kwarg.items()])


def note(test_name, description):
    """
    Add note to a test:
        lumo note <test_name> description ;

    Args:
        test_name:
        description:

    Returns:

    """
    print(f"Adding note '{description}' to {test_name}")


def board(port=11606, address=None, open=True):
    """

    Args:
        port:

    Returns:

    """
    from lumo import Watcher
    w = Watcher()
    w.panel().show(port=port, address=address, open=open)
    print(f"Starting server on port {port}")


def backup_local(test_name, target_dir, with_blob=False, with_cache=False):
    from lumo.exp.watch import Watcher
    from lumo import Experiment
    w = Watcher()
    exp = w.retrieve(test_name)
    if exp is None:
        print(f'{test_name} not found')
        exit(1)
    else:
        print(exp.backup('local', target_dir=target_dir, with_blob=with_blob, with_cache=with_cache))


def main():
    """the entry"""
    fire.Fire({
        'rerun': rerun,
        'note': note,
        'board': board,
        'backup_local': backup_local,
    })
    exit(0)
