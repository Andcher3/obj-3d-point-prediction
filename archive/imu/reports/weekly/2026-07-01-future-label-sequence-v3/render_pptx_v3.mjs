import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const artifactToolPath = require.resolve("@oai/artifact-tool");
const { FileBlob, PresentationFile } = await import(
  pathToFileURL(artifactToolPath).href
);

const deckDir =
  "D:/Pycharm/post_graduate_project/reports/weekly/2026-07-01-future-label-sequence-v3";
const pptxPath = path.join(
  deckDir,
  "2026-07-01-weekly-research-and-method-v3.pptx",
);
const outDir = path.join(deckDir, "rendered_from_pptx");

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

await fs.mkdir(outDir, { recursive: true });
const presentation = await PresentationFile.importPptx(
  await FileBlob.load(pptxPath),
);

for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await writeBlob(
    path.join(outDir, `slide_${String(index + 1).padStart(2, "0")}.png`),
    png,
  );
}

console.log(outDir);
