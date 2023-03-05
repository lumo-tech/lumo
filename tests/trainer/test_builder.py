from lumo import DatasetBuilder, DataLoaderSide


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
            .set_output_transform('xs1', lambda x: x + 1)
            .set_output_transform('ys1', lambda x: x - 1)
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

    dic = sub_builder.inputs
    assert 'xs' in dic
    assert 'ys' in dic

    dic = sub_builder.outputs
    assert 'xs1' in dic
    assert 'xs2' in dic
    assert 'ys1' in dic

    str(sub_builder)


def test_side():
    sup = create_dataset_builder()
    un = create_dataset_builder()

    dl = (
        DataLoaderSide()
            .add('sup', sup.DataLoader(batch_size=128, drop_last=True), cycle=True)
            .add('un', un.DataLoader(batch_size=32, drop_last=True))
            .zip()
    )

    assert len(dl) == len(un) // 32

    for batch in dl:
        assert isinstance(batch, dict)
        sup, un = batch['sup'], batch['un']
        assert sup

        assert sup['xs1'].shape[0] == 128
        assert 'xs1' in sup
        assert 'xs2' in sup
        assert 'ys1' in sup
        assert un['xs1'].shape[0] == 32
