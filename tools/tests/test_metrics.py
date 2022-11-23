import tools.metric as metrics
import pytest


@pytest.mark.parametrize('result, FilePath', [
    # (torch.randn((5,366,366)).numpy(),'./images/Real'), # from root of project
    ([], 'A_Wrong_Path'),
])
def test_Generate_ssim_mse(result, FilePath):
    sse_list = metrics.Generate_ssim_mse(result, FilePath)
    # assert len(sse_list) != 0
    # assert len(mse_list) != 0
    # assert len(sse_list) == len(mse_list)
    assert sse_list == "Please select the right path"
