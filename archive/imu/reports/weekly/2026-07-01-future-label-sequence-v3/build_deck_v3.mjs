import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const artifactToolPath = require.resolve("@oai/artifact-tool");
const { Presentation, PresentationFile } = await import(
  pathToFileURL(artifactToolPath).href
);

const projectRoot = "D:\\Pycharm\\post_graduate_project";
const deckDir = path.join(
  projectRoot,
  "reports",
  "weekly",
  "2026-07-01-future-label-sequence-v3",
);
const slidesDir = path.join(deckDir, "slides");
const outputPptx = path.join(
  deckDir,
  "2026-07-01-weekly-research-and-method-v3.pptx",
);
const outputMontage = path.join(deckDir, "artifact_tool_montage.webp");
const notesInspect = path.join(deckDir, "speaker_notes_inspect.ndjson");

const slideNotes = [
  "这页交代本周工作主线：前半部分补充 HAR/HAS/HAP 背景和论文调研，后半部分把切换预测转成 future patch label sequence，并形成 Transformer 方案雏形。",
  "这页给听众建立三个任务的基本区别。HAR 看当前窗口分类，HAS 看连续流中的活动段与边界，HAP/HAF 从过去观测推未来标签或未来信号。方向一同时包含识别和预测需求。",
  "这一页说明模型输入是六轴 IMU 时间序列。实际研究会把连续流切成 patch，标签可能是 window-level、patch-level、segment-level 或 future horizon。本课题还需要人体动作和手机运动两个标签对齐。",
  "这里概括方法演化。传统方法偏特征工程，深度模型直接学习波形；近期更关注连续序列、Transformer 和自监督预训练。对方向一，关键是从单窗口分类扩展到连续双标签和未来预测。",
  "这页只保留论文中可迁移的设计。SensorLM 和 SensorLLM 给出传感器语义对齐思路，EVI-MAE 提示无标签重构预训练，P2LHAP 直接影响我们对输出形式和 smoothing 的选择。",
  "P2LHAP 的关键是把连续传感器流转成 patch 标签序列。识别、分割和 forecast 都从这个序列读出。Smoothing 能去掉短时孤立错误，例如 stand 中间偶然出现一个 sit，从而减少过分割。",
  "方向一可以写成双标签未来序列预测。输入是过去 K 个 IMU patch，输出是人体动作和手机运动在当前到未来 H 个 patch 的概率。延续概率、切换时机和下一状态都从输出序列派生。",
  "结构上用 Transformer encoder 建模历史 IMU 上下文，用 parallel query decoder 一次性读取当前和未来多个位置。q0 对应当前，q1 到 qH 对应未来。decoder 不使用 causal mask，便于 future queries 之间相互建模。",
  "训练策略分两步。第一步用大量无标签 IMU 做 masked reconstruction 或 next patch prediction，学习时序结构。第二步用少量连续标注微调双标签输出。推理时对概率序列做平滑、去除短时孤立段，再读取切换点。",
];

function slideImageName(index) {
  return `slide_${String(index).padStart(2, "0")}.png`;
}

async function readImageBlob(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

const presentation = Presentation.create({
  slideSize: { width: 1600, height: 900 },
});

for (let index = 1; index <= slideNotes.length; index += 1) {
  const filePath = path.join(slidesDir, slideImageName(index));
  const slide = presentation.slides.add();
  slide.background.fill = "white";
  slide.images.add({
    blob: await readImageBlob(filePath),
    contentType: "image/png",
    alt: `weekly research report slide ${index}`,
    fit: "cover",
    position: { left: 0, top: 0, width: 1600, height: 900 },
  });
  slide.speakerNotes.textFrame.setText(slideNotes[index - 1]);
  slide.speakerNotes.setVisible(true);
}

const montage = await presentation.export({
  format: "webp",
  montage: true,
  scale: 1,
});
await writeBlob(outputMontage, montage);

const inspect = await presentation.inspect({
  kind: "slide,image,notes",
  maxChars: 20000,
});
await fs.writeFile(notesInspect, inspect.ndjson, "utf8");

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptx);
console.log(outputPptx);
