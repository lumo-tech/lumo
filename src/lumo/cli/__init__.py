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
    from lumo.exp.finder import retrieval_experiment
    exp = retrieval_experiment(test_name)
    if exp is not None:
        exp.rerun([f'--{k}={v}' for k, v in kwarg.items()])
    else:
        exit(1)


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


def server(port=8080):
    """

    Args:
        port:

    Returns:

    """
    print(f"Starting server on port {port}")


def main():
    fire.Fire({
        'rerun': rerun,
        'note': note,
        'server': server,
    })
