from dateutil.parser._parser import ParserError
from numpy.testing import assert_equal
from pandas import DataFrame, Series
from pandas.testing import assert_series_equal

from pymove.core.dataframe import MoveDataFrame


def test_has_columns():
    df = DataFrame(columns=['lat', 'lon', 'datetime'])
    assert MoveDataFrame.has_columns(df)

    df = DataFrame(columns=['lat', 'lon', 'time'])
    assert not MoveDataFrame.has_columns(df)


def test_validate_columns():
    df = DataFrame(
        data=[[0, 0, '01-01-2020', 0]],
        columns=['lat', 'lon', 'datetime', 'id']
    )
    MoveDataFrame.validate_move_data_frame(df)

    expected = Series(
        data=['float64', 'float64', '<M8[ns]', 'int64'],
        index=['lat', 'lon', 'datetime', 'id'],
        dtype='object',
        name=None,
    )
    assert_series_equal(df.dtypes, expected)

    df = DataFrame(
        data=[[0, 0]],
        columns=['lat', 'lon']
    )

    try:
        MoveDataFrame.validate_move_data_frame(df)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except KeyError:
        pass

    df = DataFrame(
        data=[['a', 0, '01-01-2020']],
        columns=['lat', 'lon', 'datetime']
    )

    try:
        MoveDataFrame.validate_move_data_frame(df)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except ValueError:
        pass

    df = DataFrame(
        data=[[0, 0, '0']],
        columns=['lat', 'lon', 'datetime']
    )

    try:
        MoveDataFrame.validate_move_data_frame(df)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except ParserError:
        pass


def test_format_labels():

    expected = {
        'col1': 'id',
        'col3': 'lon',
        'col2': 'lat',
        'col4': 'datetime'
    }
    labels = MoveDataFrame.format_labels('col1', 'col2', 'col3', 'col4')

    assert_equal(labels, expected)
