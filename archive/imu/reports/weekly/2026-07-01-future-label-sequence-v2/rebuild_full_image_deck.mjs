import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const artifactToolPath = require.resolve("@oai/artifact-tool");
const { Presentation, PresentationFile } = await import(
  pathToFileURL(artifactToolPath).href
);

const root = "D:\\Pycharm\\post_graduate_project";
const deckDir = path.join(
  root,
  "reports",
  "weekly",
  "2026-07-01-future-label-sequence-v2",
);
const slidesDir = path.join(deckDir, "slides");
const outputPptx = path.join(
  deckDir,
  "2026-07-01-weekly-research-and-method-v2.pptx",
);

function imageName(index) {
  return `slide_${String(index).padStart(2, "0")}.png`;
}

async function readImageBlob(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
}

const presentation = Presentation.create({
  slideSize: { width: 1600, height: 900 },
});

for (let index = 1; index <= 9; index += 1) {
  const filePath = path.join(slidesDir, imageName(index));
  const slide = presentation.slides.add();
  slide.background.fill = "white";
  slide.images.add({
    blob: await readImageBlob(filePath),
    contentType: "image/png",
    alt: `weekly report slide ${index}`,
    fit: "cover",
    position: { left: 0, top: 0, width: 1600, height: 900 },
  });
}

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptx);
console.log(outputPptx);
