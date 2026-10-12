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
  "2026-07-01-motion-primitive-forecast-deck",
);
const slidesDir = path.join(deckDir, "slides");
const outputPptx = path.join(
  deckDir,
  "motion-primitive-causal-seq2seq-transformer.pptx",
);
const notesInspect = path.join(deckDir, "speaker_notes_inspect.ndjson");

const slideNotes = [
  "先从相关文献出发。当前传感器序列建模大致有三条线索：一类强调传感器与语言语义对齐，一类强调无标签重构预训练，还有一类开始把连续传感器流写成未来序列预测问题。对这个任务最有直接价值的是后两点：把 IMU 转成稳定 token，并预测未来一段状态序列。",
  "这页把文献方法和结构选择对应起来。SensorLM 和 SensorLLM 说明传感器信号可以接近语义空间；EVI-MAE 提供无标签预训练思路；P2LHAP 给出 patch-level label sequence 和 smoothing；MoPFormer 给出 motion primitive tokenization；FUTR 和 ActFusion 支持并行的未来序列预测。这些共同导向右侧的结构。",
  "目标任务是基于过去到当前的手机 IMU，预测当前到未来的双流状态序列。双流分别是人体动作和手机或运镜动作，每条流都包含 action 或 motion label，以及 stable / transition phase。这样不仅能回答当前状态，也能估计未来延续、切换时间和下一状态。",
  "这一页说明数据形式。输入是连续六轴 IMU，先切成时间 patch，再在 tokenizer 内部按单通道处理。标签也必须是连续时间对齐的序列，human 和 camera 分别标注。推理阶段需要 label sequence，因为切换时机来自平滑后的第一次标签变化。",
  "方法的第一步是把单通道 IMU 片段转成 motion primitive token。归一化后的片段进入 VQ codebook，得到表示局部运动形状的 primitive embedding；均值和标准差保留为 statistical embedding；再加上通道 metadata 和时间位置编码。这样 token 同时保留局部形状和幅值信息。",
  "当前状态不需要 decoder 生成，因为时刻 t 已经在输入中。每个通道独立经过共享 causal encoder，然后在同一时间位置做 channel fusion，得到时间级 hidden state g_t。当前的 human action、human phase、camera motion 和 camera phase 都从 g_t 预测。",
  "未来状态预测分两步。第一步用 parallel query forecast decoder 预测未来 token 的动态成分，也就是 primitive embedding 和 statistical embedding；再加上已知 metadata 和未来位置编码组成完整 future motion token。第二步把真实过去 token 和预测未来 token 拼接后送入 causal re-encoder，再从未来 hidden states 做分类。",
  "训练分成两个阶段。无标签阶段只学习 IMU 动态：输入过去 IMU，预测未来 motion token，并用 primitive 和 statistical component 的对齐损失训练。微调阶段使用连续标签，分别监督 human action、human phase、camera motion 和 camera phase。这样先学时序结构，再用较少标签学习语义状态。",
  "推理阶段先得到 human 和 camera 两条概率序列，再分别进行 P2LHAP-style smoothing。平滑会去掉短时孤立错误并合并相邻同类 patch。最终当前状态读取 time t，延续趋势看未来是否保持当前标签，切换时机读取平滑序列中第一次 label 变化的位置。",
];

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

for (let index = 1; index <= slideNotes.length; index += 1) {
  const slide = presentation.slides.add();
  slide.background.fill = "white";
  slide.images.add({
    blob: await readImageBlob(path.join(slidesDir, imageName(index))),
    contentType: "image/png",
    alt: `motion primitive forecasting slide ${index}`,
    fit: "cover",
    position: { left: 0, top: 0, width: 1600, height: 900 },
  });
  slide.speakerNotes.textFrame.setText(slideNotes[index - 1]);
  slide.speakerNotes.setVisible(true);
}

const inspect = await presentation.inspect({
  kind: "slide,image,notes",
  maxChars: 30000,
});
await fs.writeFile(notesInspect, inspect.ndjson, "utf8");

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptx);
console.log(outputPptx);
