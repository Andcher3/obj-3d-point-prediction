# Project Agent Notes

ACTIVE_RESEARCH_SCOPE=VIDEO_MOTION_FORECASTING

## Project Purpose

The active project is video-only future motion forecasting for a specified object. It takes MotionForesight's future 3D point-trajectory representation as the output interface and studies two extensions:

1. represent multiple plausible futures through a probability-continuous conditional distribution;
2. use long-range, high-level semantic reasoning for global future structure while recent frames provide local motion and physical detail.

The current sources of truth are:

- `research_roadmap.md`
- `reports/ideas/2026-09-03-motionforesight-probabilistic-semantic-design.md`
- `reports/weekly/2026-08-06-flowforesight-method-slides/speaker-notes.txt`
- `reports/progress_log.md`

## Active Task Definition

Input:

- observed RGB video only;
- a specified target-object mask or equivalent target specification;
- query points sampled on that target.

Output:

- future 3D trajectories of the target points;
- a conditional distribution over multiple plausible future trajectories, not merely one regression result or a fixed number of equally weighted samples.

The point-trajectory output is retained because it avoids generating redundant appearance/background pixels while preserving object translation, rotation, articulation, deformation, direction, speed, and endpoint information.

## Working Method Hypothesis

Use a hierarchical conditional model:

```text
long-range observed context -> global semantic/intention variable
recent high-resolution frames -> local dynamic/physical state
target geometry/history       -> object and point conditions
all conditions                -> probabilistic future 3D trajectory generator
```

A useful working factorization is:

```text
p(D_future | V_obs, M_target)
= integral p(D_future | z_global, c_local, c_geometry)
           p(z_global | c_global) dz_global
```

This is a research abstraction, not a frozen architecture.

## Two Required Innovations

### 1. Probability-Continuous Future Uncertainty

- Different futures should have different probability mass or density.
- The model should support sampling while retaining a meaningful probabilistic interpretation.
- A fixed `K=6` is only a display or evaluation budget; use variable `S` for the number of samples.
- Multiple noise seeds alone demonstrate multiple samples, not calibrated uncertainty.
- Keep density estimation, mode probability, calibration, coverage, and single-future supervision as unresolved research questions until tested.

### 2. Global Semantics With Local Physical Detail

- Global/high-dimensional semantics should determine coarse future mode, interaction outcome, and overall trajectory structure.
- Short-range recent frames should determine velocity, acceleration, contact, geometry, and other fine physical details.
- Explicit VLM event hypotheses, open-vocabulary event graphs, and continuous non-text semantic latents remain candidate implementations.
- Do not present any candidate as selected or validated without evidence.

## Current Evidence And Limits

- MotionForesight and Generative Point Tracking and Forecasting have been summarized in local presentation materials.
- A conditional Flow Matching multi-future sketch and method figure exist.
- No MotionForesight reproduction, training code, dataset protocol, calibrated probability output, or semantic-guidance experiment has been completed locally.
- The existing six-output figure is stale with respect to the variable-sample interpretation.
- The FlowForesight method-slide folder currently contains images for pages 1-3; pages 4-5 remain text prompts/notes only.

## Immediate Research Order

1. Freeze the task and evaluation protocol: dataset, observed/predicted horizon, target specification, point sampling, coordinate frame, validity rules, and metrics.
2. Reproduce a single-future MotionForesight-style baseline and its pseudo-label/evaluation path.
3. Establish a no-explicit-semantics probabilistic trajectory baseline for ablation.
4. Define what probability is reported: trajectory density, neighborhood/mode mass, calibrated confidence, or another measurable quantity.
5. Compare global semantic mechanisms without mixing all sources of error at once.
6. Test whether semantic conditioning changes macro futures while recent-frame conditioning controls local dynamics.

## Archived IMU Boundary

The former IMU/HAR/OIS work is archived under `archive/imu/`. It is historical material only.

- Do not use IMU data, IMU papers, UCI HAR scripts, or IMU method designs in active reasoning, implementation, slides, or recommendations.
- Do not run archived IMU commands unless the user explicitly reactivates that direction.
- Do not mix the archived EMA Future Context/Local Motion Token ideas into the current video method solely because the names sound related.

## Literature And Evidence Rules

- Start from supplied/local papers, figures, notes, and runtime evidence.
- Separate paper facts, the user's confirmed design, current hypotheses, and new suggestions.
- Preserve the MotionForesight training/prediction boundary: future frames may produce offline pseudo-labels but do not enter prediction-time conditions.
- Retain open risks until experiments resolve them.

## PPT And Slide Generation Rule

- Generate every slide as a full-page image with the image generation tool, then insert it into the PPT unless the user explicitly overrides this rule.
- Use the existing local presentation style when available.
- Use Microsoft YaHei / 微软雅黑 styling.
- Keep slides information-dense and readable; prefer paper figures, method/data-flow diagrams, result tables, and concrete examples.
- Do not add page numbers, meaningless illustration labels, or a bottom summary bubble on every page.

## Progress Logging Rule

After each completed research, implementation, experiment, reading, or report task:

```text
reports/progress_log.md
reports/progress/YYYY-MM-DD-short-slug.md
```

Keep the active index focused on MotionForesight-based video forecasting. Archived IMU history remains under `archive/imu/`.
