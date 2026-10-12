# AGENTS workflow refresh

- 日期：2026-06-28
- 完成内容：Updated AGENTS.md with verified UCI HAR preprocessing and plotting commands, plus the dependency on regenerating data/processed/uci_har before downstream scripts.
- 本地产出：
- AGENTS.md
- reports/progress_log.md
- reports/progress/2026-06-28-agents-workflow-refresh.md
- 关键理解/结果：Future agents can now follow repo-backed commands instead of generic suggestions, and know to rerun src/data/uci_har.py before total_acc scripts if processed tensors are missing or stale.
- 下一步：Keep AGENTS.md aligned with newly added scripts or stable notebook entrypoints only after they are verified locally.
