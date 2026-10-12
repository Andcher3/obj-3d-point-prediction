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
  "D:\\Pycharm\\post_graduate_project\\reports\\weekly\\2026-07-01-motion-primitive-forecast-deck-v2";
const pptxPath = path.join(
  deckDir,
  "motion-primitive-causal-seq2seq-transformer-v3-speech-notes.pptx",
);
const outDir = path.join(deckDir, "rendered_v3");
await fs.mkdir(outDir, { recursive: true });

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const presentation = await PresentationFile.importPptx(
  await FileBlob.load(pptxPath),
);

for (const [index, slide] of presentation.slides.items.entries()) {
  const rendered = await presentation.export({ slide, format: "png", scale: 1 });
  await writeBlob(
    path.join(outDir, `slide-${String(index + 1).padStart(2, "0")}.png`),
    rendered,
  );
}

const inspect = await presentation.inspect({
  kind: "slide,image,notes",
  maxChars: 80000,
});
await fs.writeFile(path.join(deckDir, "pptx_render_inspect_v3.ndjson"), inspect.ndjson, "utf8");

console.log(outDir);
