import numpy as np
import polars as pl

import fisseq_data_pipeline.harmonize as mod


class StubConfig:
    def __init__(self, cfg):
        # Provide the attribute used by the module
        # you can override in tests by passing a dict with this key if desired
        self.batch_col_name = (
            getattr(cfg, "batch_col_name", "batch")
            if hasattr(cfg, "__dict__")
            else "batch"
        )


def test_fit_harmonizer_calls_harmonizationLearn_with_expected_args(monkeypatch):
    """
    Ensures:
      - Config() is invoked
      - get_control_samples is used
      - get_feature_matrix provides the matrix used to fit
      - covariates are collected from control_df's batch column and passed as pandas with SITE alias
      - return value is exactly the model returned by harmonizationLearn
    """
    # Arrange
    monkeypatch.setattr(mod, "Config", StubConfig, raising=True)

    # control DF with batch column; feature columns won't actually be read by our stub
    control_df_pd = pl.DataFrame(
        {
            "batch": ["A", "A", "B", "B"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
        }
    )
    control_lf = control_df_pd.lazy()

    # The train_data_df passed in; our get_control_samples returns control_lf regardless
    train_lf = pl.DataFrame({"batch": ["Z"], "f1": [0.0], "f2": [0.0]}).lazy()

    # Stub get_control_samples to return our control_lf
    calls = {"get_control_samples": 0, "get_feature_matrix": 0, "learn": 0}

    def fake_get_control_samples(lf, cfg):
        calls["get_control_samples"] += 1
        assert isinstance(lf, pl.LazyFrame)
        assert isinstance(cfg, StubConfig)
        return control_lf

    # Stub get_feature_matrix to return features for fitting
    control_mat = np.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=float
    )

    def fake_get_feature_matrix(lf, cfg):
        calls["get_feature_matrix"] += 1
        # Should be called with the control_lf
        assert lf is control_lf
        return (["f1", "f2"], control_mat)

    # Capture args to neuroHarmonize.harmonizationLearn and return a sentinel model
    def fake_harmonizationLearn(X, covar_df_pd):
        calls["learn"] += 1
        # X should be the control feature matrix we returned
        assert isinstance(X, np.ndarray)
        np.testing.assert_allclose(X, control_mat)
        # Covariate should be a pandas DF with a single 'SITE' column mapped from control 'batch'
        assert list(covar_df_pd.columns) == ["SITE"]
        assert covar_df_pd["SITE"].tolist() == control_df_pd["batch"].to_list()
        return ({"model_key": "MODEL_OK"}, {"meta": "ignored"})

    monkeypatch.setattr(
        mod, "get_control_samples", fake_get_control_samples, raising=True
    )
    monkeypatch.setattr(
        mod, "get_feature_matrix", fake_get_feature_matrix, raising=True
    )
    monkeypatch.setattr(
        mod.neuroHarmonize, "harmonizationLearn", fake_harmonizationLearn, raising=True
    )

    # Act
    model = mod.fit_harmonizer(train_lf, config={})

    # Assert
    assert calls["get_control_samples"] == 1
    assert calls["get_feature_matrix"] == 1
    assert calls["learn"] == 1
    assert model == {"model_key": "MODEL_OK"}


def test_harmonize_calls_harmonizationApply_and_sets_features(monkeypatch):
    """
    Ensures:
      - get_feature_matrix provides (cols, X)
      - covariates derived from data_df's batch column are passed as pandas with SITE alias
      - neuroHarmonize.harmonizationApply output is handed to set_feature_matrix
      - return value matches what set_feature_matrix returns
    """
    monkeypatch.setattr(mod, "Config", StubConfig, raising=True)

    # Source data (lazy)
    df = pl.DataFrame(
        {
            "batch": ["A", "A", "B"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [10.0, 20.0, 30.0],
            "other": [7, 8, 9],
        }
    )
    data_lf = df.lazy()

    # Feature matrix returned by get_feature_matrix
    cols = ["f1", "f2"]
    X = np.column_stack([df["f1"].to_numpy(), df["f2"].to_numpy()])  # shape (3, 2)
    calls = {"get_feature_matrix": 0, "apply": 0, "set_feature_matrix": 0}

    def fake_get_feature_matrix(lf, cfg):
        calls["get_feature_matrix"] += 1
        assert lf is data_lf
        return cols, X.copy()

    # Harmonizer object can be arbitrary (a dict)
    harmonizer = {"model_key": "MODEL_OK"}

    # neuroHarmonize.harmonizationApply returns a transformed matrix; just add 1 for test
    def fake_harmonizationApply(X_in, covar_pandas, model):
        calls["apply"] += 1
        assert model is harmonizer
        # Check covariates are a pandas DF with SITE column from data_lf['batch']
        assert list(covar_pandas.columns) == ["SITE"]
        assert covar_pandas["SITE"].tolist() == df["batch"].to_list()
        return X_in + 1.0

    # set_feature_matrix should be called with the transformed matrix
    captured = {}

    def fake_set_feature_matrix(lf, feature_cols, new_features):
        calls["set_feature_matrix"] += 1
        captured["lf"] = lf
        captured["feature_cols"] = feature_cols
        captured["new_features"] = new_features
        # Return a sentinel LazyFrame as the function result
        return pl.DataFrame(
            {"f1": new_features[:, 0], "f2": new_features[:, 1], "other": df["other"]}
        ).lazy()

    monkeypatch.setattr(
        mod, "get_feature_matrix", fake_get_feature_matrix, raising=True
    )
    monkeypatch.setattr(
        mod.neuroHarmonize, "harmonizationApply", fake_harmonizationApply, raising=True
    )
    monkeypatch.setattr(
        mod, "set_feature_matrix", fake_set_feature_matrix, raising=True
    )

    # Act
    out_lf = mod.harmonize(data_lf, config={}, harmonizer=harmonizer)
    out = out_lf.collect()

    # Assert call flow and arguments
    assert calls["get_feature_matrix"] == 1
    assert calls["apply"] == 1
    assert calls["set_feature_matrix"] == 1
    np.testing.assert_allclose(captured["new_features"], X + 1.0)
    assert captured["feature_cols"] == cols

    # Returned LF should reflect transformed features (X + 1)
    np.testing.assert_allclose(out["f1"].to_numpy(), (df["f1"].to_numpy() + 1.0))
    np.testing.assert_allclose(out["f2"].to_numpy(), (df["f2"].to_numpy() + 1.0))
    assert out["other"].to_list() == df["other"].to_list()


def test_harmonize_preserves_feature_order(monkeypatch):
    """
    If get_feature_matrix returns columns in a non-standard order,
    ensure that order is propagated through set_feature_matrix.
    """
    monkeypatch.setattr(mod, "Config", StubConfig, raising=True)

    df = pl.DataFrame({"batch": ["A", "B"], "a": [1.0, 2.0], "b": [10.0, 20.0]}).lazy()
    cols = ["b", "a"]
    X = np.array([[10.0, 1.0], [20.0, 2.0]], dtype=float)

    def fake_get_feature_matrix(lf, cfg):
        return cols, X.copy()

    def fake_apply(X_in, covar_pd, model):
        # Identity transform for this test
        return X_in

    seen = {}

    def fake_set_feature_matrix(lf, feature_cols, new_features):
        seen["cols"] = feature_cols[:]  # copy
        # Return something that we can collect
        return pl.DataFrame({"b": new_features[:, 0], "a": new_features[:, 1]}).lazy()

    monkeypatch.setattr(
        mod, "get_feature_matrix", fake_get_feature_matrix, raising=True
    )
    monkeypatch.setattr(
        mod.neuroHarmonize, "harmonizationApply", fake_apply, raising=True
    )
    monkeypatch.setattr(
        mod, "set_feature_matrix", fake_set_feature_matrix, raising=True
    )

    out_lf = mod.harmonize(df, config={}, harmonizer={"m": 1})
    _ = out_lf.collect()

    assert seen["cols"] == cols  # order preserved
