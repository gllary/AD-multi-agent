# LightGBM 工具权重

目录结构::

  lgbm/
    CP1/fold_models/fold*_booster.txt
    CP2/fold_models/fold*_booster.txt
    CP2E/fold_models/fold*_booster.txt

本仓库已随 ``artifacts/models/lgbm/`` 附带与 **datasetA 训练** 一致的 5 折模型，可直接用于推断。

若你重新训练了模型，覆盖上述 ``fold*_booster.txt`` 即可。
