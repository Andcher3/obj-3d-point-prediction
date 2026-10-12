# Directional movement repair

- Source: `references/running-right-source.gif` (276x208, 18 frames, 30 ms per frame).
- The source contains six distinct poses, each held for three GIF frames.
- Codex v2 directional rows require exactly eight atlas cells. The rightward row uses source frame indices `0, 2, 5, 7, 9, 11, 14, 16`, preserving the source order and covering all six distinct poses.
- The character was not redrawn or regenerated. Processing was limited to connected white-background removal, uniform downscaling into a 192x208 cell, and transparent padding.
- The leftward row was produced by framewise horizontal mirroring with frame order preserved.
- Rows 0 and 3-10 were preserved from the previously approved v2 atlas.
- Final atlas validation: 1536x2288 RGBA WebP, sprite version 2, no transparent RGB residue, no chroma leak, no warnings or errors.
