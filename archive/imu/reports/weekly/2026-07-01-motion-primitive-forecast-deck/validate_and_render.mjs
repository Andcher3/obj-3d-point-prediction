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
  "D:/Pycharm/post_graduate_project/reports/weekly/2026-07-01-motion-primitive-forecast-deck";
const pptxPath = path.join(deckDir, "motion-primitive-causal-seq2seq-transformer.pptx");
const renderDir = path.join(deckDir, "rendered_from_pptx");
const inspectPath = path.join(deckDir, "final_inspect.ndjson");

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

const presentation = await PresentationFile.importPptx(
  await FileBlob.load(pptxPath),
);
const inspect = await presentation.inspect({
  kind: "slide,image,notes",
  maxChars: 30000,
});
await fs.writeFile(inspectPath, inspect.ndjson, "utf8");

const rows = inspect.ndjson.trim().split(/\n/).filter(Boolean).map(JSON.parse);
const slides = rows.filter((row) => row.kind === "slide");
const images = rows.filter((row) => row.kind === "image");
const notes = rows.filter(
  (row) => row.kind === "notes" && row.text && row.text.trim().length > 0,
);

await fs.mkdir(renderDir, { recursive: true });
for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await writeBlob(
    path.join(renderDir, `slide_${String(index + 1).padStart(2, "0")}.png`),
    png,
  );
}

console.log(JSON.stringify({
  slides: slides.length,
  images: images.length,
  notes: notes.length,
  renderDir,
}, null, 2));
