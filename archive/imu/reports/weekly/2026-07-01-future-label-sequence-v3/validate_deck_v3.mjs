import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const artifactToolPath = require.resolve("@oai/artifact-tool");
const { FileBlob, PresentationFile } = await import(
  pathToFileURL(artifactToolPath).href
);

const pptxPath =
  "D:/Pycharm/post_graduate_project/reports/weekly/2026-07-01-future-label-sequence-v3/2026-07-01-weekly-research-and-method-v3.pptx";

const presentation = await PresentationFile.importPptx(
  await FileBlob.load(pptxPath),
);
const inspect = await presentation.inspect({
  kind: "slide,notes,image",
  maxChars: 30000,
});
const rows = inspect.ndjson.trim().split(/\n/).filter(Boolean).map(JSON.parse);
const slides = rows.filter((row) => row.kind === "slide");
const images = rows.filter((row) => row.kind === "image");
const notes = rows.filter(
  (row) => row.kind === "notes" && row.text && row.text.trim().length > 0,
);

console.log(JSON.stringify({
  slides: slides.length,
  images: images.length,
  notes: notes.length,
}, null, 2));
for (const [index, note] of notes.entries()) {
  console.log(`${index + 1}: ${note.text.slice(0, 48)}`);
}
