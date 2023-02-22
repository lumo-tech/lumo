from lumo import DatasetBuilder


def global_check(dic):
    for item in ['xs1', 'xs1', 'ys1']:
        assert item in dic
    return dic


def create_dataset_builder():
    builder = (
        DatasetBuilder()
            .add_input(name='xs', source=range(1000))
            .add_input(name='ys', source=range(1, 1001))
            .add_output(name='xs', outkey='xs1')
            .add_output(name='xs', outkey='xs2')
            .add_output(name='ys', outkey='ys1')
            .add_output_transform('xs1', lambda x: x + 1)
            .add_output_transform('ys1', lambda x: x - 1)
            .add_global_transform(global_check)
    )
    return builder


def test_builder_base():
    builder = create_dataset_builder()

    try:
        builder.add_input('xs', source=range(1000))
        assert False
    except:
        pass
    try:
        builder.add_input('xxs', source=range(1001))
        assert False
    except:
        pass

    assert len(builder) == 1000

    sub_builder = builder.subset(range(500), copy=True)
    assert len(builder) == 1000
    assert len(sub_builder) == 500
    assert sub_builder[499]['xs1'] == 500
    assert sub_builder[499]['ys1'] == 499
    assert sub_builder[499]['xs2'] == 499
