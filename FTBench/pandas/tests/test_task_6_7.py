import FTBench.pandas.T6_pandas as t6, FTBench.pandas.T7_pandas as t7

def test_task_6():
    # run benchmark, including read and prep (via T6.py)
    result_df = t6.benchmark_t6()

    # test assertions
    assert result_df.shape == (48000000, 10), f"Expected (48000000, 10) but got {result_df.shape}"


def test_task_7():
    # run benchmark, including read and prep (via T6.py)
    result_df = t7.benchmark_t7()

    # test assertions
    assert result_df.shape == (48000000, 10), f"Expected (48000000, 10) but got {result_df.shape}"