import tools.visualisation as visualisation
import pytest


@pytest.mark.parametrize('x, rows, cols, res', [
    (5, 1, 1, "dataset is not in correct format please insert a dictionary"),
])
def test_show_batch(x, rows, cols, res):
    """ Test the update_prediction function """
    result = visualisation.show_batch(x, rows, cols)
    assert result == res
