import T6 as t6, T7 as t7

def test_task_6():
    # run benchmark, including read and prep (via T6.py)
    result_df = t6.benchmark_t6()

    # test assertions
    assert result_df.shape == (48000000, 10), f"Expected (48000000, 10) but got {result_df.shape}"


def test_task_7():
    # run benchmark, including read and prep (via T6.py)
    result_df = t7.benchmark_t6()

    # test assertions
    assert result_df.shape == (48000000, 10), f"Expected (48000000, 10) but got {result_df.shape}"