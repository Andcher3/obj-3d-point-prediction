# WorldTrack metric reference snapshot

Source: https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/track_eval_util.py

Retrieved 2026-10-10. Text snapshot for non-commercial scientific verification,
consistent with the upstream README use statement. Retained separately from our
implementation; not imported as a complete module. Tests extract only three named
metric functions via AST and use NumPy arrays plus a non-Torch type placeholder.
No upstream model, Torch, GPU, optimization or inference is executed by these tests.

The global median alignment, four inclusive thresholds and EPE are differentially
checked on 120 finite synthetic cases. This does not validate official data selection,
UniQuery4R's unpublished evaluation wrapper or model performance. Input selection and
dynamic classification were separately read from the pinned track_eval.py source.
