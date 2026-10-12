# Project Agent Notes

## Project Purpose

This workspace is for summer preparation before starting graduate research in artificial intelligence. The user is an AI major at the beginning of graduate school and wants to build enough background to start later research work confidently.

The source of truth for the possible research directions is:

- `requirements/课题汇总.pdf`
- `research_roadmap.md`

The current priority is to understand the first two IMU-based topics before expanding into the third video reconstruction topic.

## Research Directions

### 1. Mobile Human Activity And Camera Motion Recognition

Goal: use smartphone IMU signals to jointly model, recognize, and predict human behavior and phone camera-motion behavior.

Important concepts:

- accelerometer and gyroscope time series
- human activity recognition
- camera motion recognition
- composite behavior modeling
- behavior state transition prediction
- lightweight real-time mobile inference

This direction is more about semantic understanding of motion.

### 2. Short-Term Pose Regression And Impact Event Prediction For OIS

Goal: use recent smartphone IMU history to predict short-term future pose or IMU trajectories, and warn about impact events for downstream optical image stabilization control.

Important concepts:

- short-horizon time series prediction
- gyro/acc regression
- impact event detection
- arrival-time regression
- multi-task learning
- streaming inference and model compression

This direction is more about predictive control support.

### 3. High-Quality Reconstruction Of Degraded Video

Goal: detect degraded regions in mobile video and reconstruct clearer, more coherent video using RGB frames and possibly sensor information.

Important concepts:

- video degradation detection
- video frame interpolation
- video deblurring
- visual odometry
- 3D Gaussian Splatting
- lightweight deployment

This is currently lower priority than the first two directions.

## Current Stage

The current active stage is Stage 1 from `research_roadmap.md`: understanding IMU data and signal structure through the UCI HAR dataset.

The dataset has been extracted under:

- `HAR/UCI HAR Dataset/`

The original archive is still present:

- `human+activity+recognition+using+smartphones.zip`

## UCI HAR Dataset Structure

Each row in the dataset represents one sliding time window sample.

Important constants:

- sampling rate: 50 Hz
- window length: 2.56 seconds
- readings per window: 128
- activities: 6
- subjects: 30

Activities:

```text
1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING
```

### Main Files

For `train/` and `test/`, the structures are equivalent.

`X_train.txt` / `X_test.txt`:

- rows are window samples
- columns are 561 precomputed statistical and frequency-domain features
- use these for traditional machine learning baselines

`y_train.txt` / `y_test.txt`:

- rows align with `X_*`
- each row is the activity label for that sample

`subject_train.txt` / `subject_test.txt`:

- rows align with `X_*`
- each row is the subject ID for that sample
- useful for subject-wise generalization analysis

### Inertial Signals

The `Inertial Signals/` folders contain the time series data that should be used first in Stage 1.

Each file has:

- rows: window samples
- columns: 128 time points inside that window

The 9 channels are:

```text
total_acc_x
total_acc_y
total_acc_z
body_acc_x
body_acc_y
body_acc_z
body_gyro_x
body_gyro_y
body_gyro_z
```

Meanings:

- `total_acc_*`: total accelerometer signal, including body motion and gravity, in g units
- `body_acc_*`: body acceleration after gravity separation, in g units
- `body_gyro_*`: gyroscope angular velocity, in rad/s

A single sample can be organized as either:

```text
9 x 128
```

or, more conveniently for deep learning:

```text
128 x 9
```

## Recommended Immediate Work

First focus on understanding the raw IMU sequence data rather than jumping directly into large models.

Recommended next steps:

1. Load the 9 `Inertial Signals` channels.
2. Join them with `y_train.txt` and `activity_labels.txt`.
3. Plot several windows from each activity class.
4. Compare periodic motion classes with static classes.
5. Then build a simple HAR baseline.

Useful first outputs:

- `notebooks/01_uci_har_eda.ipynb`
- `src/data/uci_har.py`
- `reports/uci_har_data_notes.md`

## Confirmed Local Workflow

The workspace already has a small, working UCI HAR preprocessing and visualization flow.

Use these commands from the project root:

```powershell
python .\src\data\uci_har.py
python .\scripts\plot_uci_har_samples.py
python .\scripts\plot_uci_har_total_acc_summary.py
python .\scripts\summarize_uci_har_total_acc.py
```

What they do:

- `python .\src\data\uci_har.py`
  - writes `data/processed/uci_har/X_train_raw_float32.bin`
  - writes `data/processed/uci_har/shape.json`
  - writes aligned metadata files such as `train_metadata.csv` and `activity_labels.csv`
- `python .\scripts\plot_uci_har_samples.py`
  - generates per-activity sample SVGs and `reports/figures/uci_har/index.html`
- `python .\scripts\plot_uci_har_total_acc_summary.py`
  - generates `reports/figures/uci_har/total_acc_summary.svg`
- `python .\scripts\summarize_uci_har_total_acc.py`
  - prints class-level `total_acc` summary statistics to the terminal

Important dependency:

- The two `total_acc` scripts and the sample-gallery script rely on the processed train tensor under `data/processed/uci_har/`.
- If `X_train_raw_float32.bin` is missing or stale, rerun `python .\src\data\uci_har.py` first.

## Confirmed Local Weekly Material Scripts

The workspace also contains a small number of topic-specific material-generation scripts for archived weekly outputs.

Use these commands from the project root when you need to regenerate the same materials:

```powershell
python .\scripts\create_sensorlm_weekly_materials.py
python .\scripts\create_future_label_sequence_materials.py
```

What they do:

- `python .\scripts\create_sensorlm_weekly_materials.py`
  - writes 4 PNG slide images under `reports/weekly/sensorlm_slides/`
  - this is a SensorLM-specific slide-image script, not a general presentation entrypoint
- `python .\scripts\create_future_label_sequence_materials.py`
  - writes `reports/posters/2026-06-30-future-label-sequence-method-poster.png`
  - writes `reports/posters/2026-06-30-future-label-sequence-method-poster.pdf`
  - writes `reports/weekly/2026-07-01-future-label-sequence/2026-07-01-future-label-sequence-method.pptx`
  - this is a dated script for that specific poster/PPT bundle, not a general slide workflow

Important dependency:

- Both scripts use `Pillow`.
- `create_future_label_sequence_materials.py` also uses `python-pptx`.

## Confirmed Archived Deck Build Commands

Several date-stamped weekly deck folders also contain local image-per-slide PPT build/inspection scripts. These are archived, folder-scoped workflows rather than general entrypoints under `scripts/`.

Use these commands from the project root only when you need to rebuild the same archived decks:

```powershell
node .\reports\weekly\2026-07-01-future-label-sequence-v3\build_deck_v3.mjs
node .\reports\weekly\2026-07-01-future-label-sequence-v3\validate_deck_v3.mjs
node .\reports\weekly\2026-07-01-future-label-sequence-v3\render_pptx_v3.mjs
node .\reports\weekly\2026-07-01-motion-primitive-forecast-deck\build_deck.mjs
node .\reports\weekly\2026-07-01-motion-primitive-forecast-deck\validate_and_render.mjs
node .\reports\weekly\2026-07-01-motion-primitive-forecast-deck-v2\build_deck_v2.mjs
node .\reports\weekly\2026-07-01-motion-primitive-forecast-deck-v2\render_final_deck_v2.mjs
node .\reports\weekly\2026-07-01-motion-primitive-forecast-deck-v2\build_deck_v3_speech_notes.mjs
node .\reports\weekly\2026-07-01-motion-primitive-forecast-deck-v2\render_final_deck_v3.mjs
```

What they do:

- `2026-07-01-future-label-sequence-v3/*.mjs`
  - rebuilds `2026-07-01-weekly-research-and-method-v3.pptx`
  - inspects slide/image/notes counts into `speaker_notes_inspect.ndjson`
  - renders the final PPTX back into `rendered_from_pptx/`
- `2026-07-01-motion-primitive-forecast-deck/*.mjs`
  - rebuilds `motion-primitive-causal-seq2seq-transformer.pptx`
  - inspects notes into `speaker_notes_inspect.ndjson`
  - renders the final PPTX back into `rendered_from_pptx/` and `final_inspect.ndjson`
- `2026-07-01-motion-primitive-forecast-deck-v2/*.mjs`
  - rebuilds the v2/v3-speech-notes PPTX variants in that folder
  - writes speaker-note inspection files such as `speaker_notes_inspect_v2.ndjson` and `speaker_notes_inspect_v3.ndjson`
  - renders final deck previews such as `rendered/`, `rendered_v3/`, and `pptx_render_montage_v2.webp`

Important dependency:

- These archived `.mjs` scripts resolve `@oai/artifact-tool`.
- Node.js is present locally, but a plain `node ...` run currently fails unless `@oai/artifact-tool` is available in the active environment.
- TODO: if a stable repo-level setup for `@oai/artifact-tool` is added, document that setup before treating these archived deck commands as a general reusable workflow.

Additional verified scope note:

- The newer weekly folders `reports/weekly/2026-07-05-slide-image-edits/`, `reports/weekly/2026-07-07-ema-future-context-method-slides/`, and `reports/weekly/2026-07-08-ema-target-task-slide/` currently contain slide-image artifacts only.
- No repo-local rebuild script or command entrypoint is currently verified for those folders.
- TODO: if a reproducible export or assembly command for these newer slide-image folders is added later, document that command instead of treating the folders themselves as runnable workflows.

## Literature Reading Sources

The user usually reads and manages papers in Zotero. When continuing paper reading in this project, prefer local Zotero PDF files when the user provides a storage path, and use web sources mainly for discovery, citation metadata, or missing PDFs.

Current local Zotero PDF folder for the first HAR review:

- `D:\Zotero_file\storage\9VNZFD92`

This folder contains the local PDF for:

- `Deep Learning in Human Activity Recognition with Wearable Sensors: A Review on Advances`

When reading a local paper PDF, extract or inspect the PDF directly and ground summaries in the local file whenever practical.

## Working Style For Future Agents

- Explain concepts at a beginner-friendly graduate-student level.
- Prefer concrete local files and dataset evidence over generic summaries.
- Keep the current focus on IMU fundamentals unless the user explicitly switches topics.
- Do not jump to the third video reconstruction topic unless asked.
- When describing matrix shapes, be precise about rows and columns.
- Treat `test/` as the official test split, not a validation set. If validation is needed, split it from `train/`, preferably subject-aware.
- For code work, prefer small, inspectable scripts or notebooks before larger abstractions.
- Avoid overwriting user-created notes or experiments.
- For UCI HAR plotting/statistics work, check whether `data/processed/uci_har/X_train_raw_float32.bin` and `shape.json` exist before running downstream scripts.

## PPT And Slide Generation Rule

When making any PPT, slide deck, paper presentation, or group-meeting slides in this project, follow these user requirements:

- Each slide must be generated as a full-page image using the image generation skill/tool, then inserted into the PPT. Do not build slides mainly from editable text boxes unless the user explicitly overrides this rule.
- For main content slides, use the existing local PPT style as the visual template whenever available.
- Keep the deck information-dense but readable: layouts should not feel sparse, and they should not feel overcrowded.
- Use Microsoft YaHei / 微软雅黑 as the primary Chinese and English font style in generated slide images.
- Use vector-style diagrams only when they genuinely help comprehension. Do not overuse vector art. If a diagram cannot make the audience infer the matching concept without reading nearby text, do not include it.
- Prefer real examples, paper-style figures, model/data-flow diagrams, result tables, and academically useful illustrations as slide visuals.
- Do not add meaningless labels such as "real scene example" or "academic style illustration" to explain what an inserted image is. The title/caption should describe the actual content.
- Do not put a bottom summary bubble on every slide. Add a summary callout only when the slide genuinely benefits from one.
- Do not show page numbers on slides unless the user explicitly asks for them.
- Do not write a section's functional role directly as its local heading. For example, avoid headings like "评价指标解释"; use layout, grouping, typography, or concise content titles to express the role visually.

## Progress Logging Rule

The user needs local records for a weekly group meeting, usually on Wednesday. After each completed task, create one per-item progress file and update the summary/index.

Use this structure:

```text
reports/
  progress_log.md
  progress/
    YYYY-MM-DD-short-slug.md
```

`reports/progress_log.md` should remain a concise overview and index. Detailed notes belong in separate files under `reports/progress/`.

Each entry should be short and useful for reporting:

- date
- completed work
- local outputs or changed files
- key understanding or result
- suggested next step

Use the `research-progress-logger` skill when available. Do this as part of the task close-out, not as a separate optional follow-up.

## Suggested Mental Model

For the current UCI HAR work:

```text
Inertial Signals/*.txt  -> raw-ish IMU time series windows
X_train/X_test.txt      -> precomputed 561-dimensional feature table
y_train/y_test.txt      -> activity labels
subject_train/test.txt  -> subject IDs
```

The key learning goal is to connect visible IMU waveforms with human activities before moving on to classification, transition modeling, and short-term prediction.
