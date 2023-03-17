import pandas as pd
from lumo import C

# create a sample DataFrame
data = {'name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
        'age': [25, 30, 35, 40, 45],
        'city': [{'index': 0}, {'index': 1}, {'index': 2}, {'index': 3}, {'index': 4}]}
df = pd.DataFrame(data)


def test_row_filter():
    # create and apply the condition to filter the DataFrame
    filtered_df = (C['age'] >= 35).apply(df)
    # assert that only the rows with age >= 35 are in the filtered DataFrame
    assert filtered_df.shape == (3, 3)
    assert filtered_df['age'].min() >= 35


def test_column_edit_add():
    # test the column edit operation C+{'city.index':'cindex'}
    assert (C + {'city.index': 'cindex'}).op == 'add_column'
    new_df = (C + {'city.index': 'cindex'}).apply(df)
    assert 'cindex' in new_df.columns
    assert new_df.columns.tolist() == ['name', 'age', 'city', 'cindex']

    # test the column edit operation C-['city']
    new_df = (C - ['city']).apply(df)
    assert 'city' not in new_df.columns
    assert new_df.columns.tolist() == ['name', 'age']

    # test the column edit operation C-['city','name']
    new_df = (C - ['city', 'name']).apply(df)
    assert 'city' not in new_df.columns
    assert 'name' not in new_df.columns
    assert new_df.columns.tolist() == ['age']


def test_pipeline():
    # test the pipeline operation
    filtered_df = C.pipe(df, [
        (C['age'] > 35),
        C + {'city.index': 'cindex'},
        C - ['city', 'name']
    ])
    assert filtered_df.shape == (2, 2)
    assert 'name' not in filtered_df.columns
    assert 'city' not in filtered_df.columns
    assert 'age' in filtered_df.columns
    assert 'cindex' in filtered_df.columns
