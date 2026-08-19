# 训练产物归档

训练运行不是源码。每个运行应携带一个 `run.json`，至少记录：

- `product_id`、`product_version` 和 `task_id`；
- 配置、资产、payload 和源码版本摘要；
- 运行目录、状态和时间；
- `parent_run_id`、`parent_checkpoint` resume 父链；
- 诊断、遥测和其他可查询元数据。

`TrainingArchive` 将这些记录写入 SQLite 索引，checkpoint、日志和回放仍留在运行目录，
不复制到源码包。默认索引目录为 `output/training_archive/`，可用
`LOCOMOTION_TRAINING_ARCHIVE_ROOT` 覆盖。
