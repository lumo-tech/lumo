from lumo import DatasetBuilder, DataLoaderSide, DataModule, ParamsType, TrainStage


def global_check(dic):
    for item in ['xs1', 'xs1', 'ys1']:
        assert item in dic
    return dic


def create_dataset_builder():
    builder = (
        DatasetBuilder()
            .add_idx('id')
            .add_input(name='xs', source=range(1000))
            .add_input(name='axs', source=range(1000), transform=lambda x: x - 1)
            .add_input(name='ys', source=range(1, 1001))
            .add_output(name='xs', outkey='xs1', transform=lambda x: x + 1)
            .add_output(name='xs', outkey='xs2')
            .add_output(name='axs', outkey='xs3')
            .add_output(name='ys', outkey='ys1')
            .set_output_transform('ys1', lambda x: x - 1)
            .add_global_transform(global_check)
    )
    return builder


def test_iter_data():
    builder = (
        DatasetBuilder()
            .add_idx('id')
            .add_input('xs', iter(range(20)))
            .add_input('ys', iter(range(20)))
            .add_output('xs', 'xs1')
            .add_output('xs', 'xs2', transform=lambda x: x + 1)
            .add_output('ys', 'ys')
    )
    try:
        builder[0]
        assert False
    except TypeError:
        assert True

    try:
        len(builder)
        assert False
    except TypeError:
        assert True

    for i, sample in enumerate(builder):
        assert isinstance(sample, dict)
        assert sample['xs1'] == i
        assert sample['xs2'] == i + 1
        assert sample['ys'] == i
        assert sample['id'] == i

    builder.chain()
    for i, (xs1, xs2, ys) in enumerate(builder):
        assert xs1 == i
        assert xs2 == i + 1
        assert ys == i


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

    sub_builder = builder.subset(range(20), copy=True)
    assert len(builder) == 1000
    assert builder != sub_builder
    new_builder = builder.subset(range(500))
    assert len(builder) == 500
    assert builder == new_builder

    assert len(sub_builder) == 20

    for i, sample in enumerate(sub_builder):
        assert sample['id'] == i
        assert sample['xs1'] == i + 1
        assert sample['ys1'] == i
        assert sample['xs2'] == i
        assert sample['xs3'] == i - 1

    dic = sub_builder.inputs
    assert 'xs' in dic
    assert 'ys' in dic

    dic = sub_builder.outputs
    assert 'xs1' in dic
    assert 'xs2' in dic
    assert 'ys1' in dic

    str(sub_builder)

    sub_builder.zip()
    assert isinstance(sub_builder[0], dict)
    sub_builder.chain()
    assert isinstance(sub_builder[0], list)
    sub_builder.item()
    print(sub_builder[0])
    # assert isinstance(sub_builder[0], dict)


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


class MyDataModule(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        super().idataloader(params, stage)
        sup = create_dataset_builder()
        un = create_dataset_builder()

        dl = (
            DataLoaderSide()
                .add('sup', sup.DataLoader(batch_size=128, drop_last=True), cycle=True)
                .add('un', un.DataLoader(batch_size=32, drop_last=True))
                .zip()
        )
        self.regist_dataloader_with_stage(stage, dl)


def test_dm_dataloader():
    dm = MyDataModule()
    loader = dm.train_dataloader
    assert dm.train_dataset == loader.dataset
    assert isinstance(dm.train_dataloader, DataLoaderSide)
