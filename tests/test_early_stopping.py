from early_stopping_pytorch import EarlyStopping


def test_earlystopping(mocker):
    mocker.patch.object(EarlyStopping, "save_checkpoint")
    model = mocker.Mock()
    early_stopping = EarlyStopping()

    early_stopping(0.1, model)
    assert early_stopping.best_score == -0.1

    early_stopping(0.2, model)
    assert early_stopping.best_score == -0.1

    early_stopping(0.05, model)
    assert early_stopping.best_score == -0.05
