from lumo.kit.builder import DatasetBuilder


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
    assert len(builder) == 1000 and len(sub_builder) == 500

    assert sub_builder[499]['xs1'] == 500
    assert sub_builder[499]['ys1'] == 499
    assert sub_builder[499]['xs2'] == 499
