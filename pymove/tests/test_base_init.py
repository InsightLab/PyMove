try:
    from pymove import *  # noqa

    _top_import_error = None
except Exception as e:
    _top_import_error = e


def test_import_skl():
    assert _top_import_error is None
