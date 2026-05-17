
from src.gmf.train import run_train
from src.gmf.eval import run_eval

def test_train_eval_runs(tmp_path):
    tr = run_train(benchmark="associative_recall", model_name="gmf", dim=16, memory_cells=32, steps=5, batch_size=4, out_dir=str(tmp_path))
    ckpt = f'{tr["run_dir"]}/checkpoint.pt'
    res = run_eval(benchmark="associative_recall", model_name="gmf", ckpt=ckpt, dim=16, memory_cells=32, batches=2, batch_size=4, out_dir=str(tmp_path))
    assert "cosine_mean" in res["agg"]
