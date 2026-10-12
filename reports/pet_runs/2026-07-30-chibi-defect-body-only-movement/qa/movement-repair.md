# Directional movement repair

- Source: `references/movement-source.gif` (18 frames at 30 ms per frame).
- The five pale-blue shield/wing components were removed by retaining only the connected component containing the yellow robot head.
- The eight Codex row cells use source frame indices `0, 2, 5, 7, 9, 11, 14, 16`, preserving the source pose order and all six distinct poses.
- All selected source frames share one union crop and one scale factor, preserving the source's relative bobbing while avoiding per-frame size popping.
- The scaled union is 188x142 inside each 192x208 cell.
- Direction mapping was corrected: the original GIF orientation is `running-left` (head at screen-left), while its framewise mirror is `running-right` (head at screen-right).
- The yellow character area is approximately 5.0x larger than in the preceding winged version.
- Rows 0 and 3-10 are preserved from the previously approved v2 pet.
