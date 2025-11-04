const ort = require('onnxruntime-node');
const { createCanvas, loadImage } = require('canvas');

async function runYOLO() {
  const session = await ort.InferenceSession.create('./yolov8n.onnx');

  const image = await loadImage('./chip.jpg');
  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);

  let inputTensor = tf.browser.fromPixels(canvas).expandDims(0).toFloat().div(255); // 或使用 onnxruntime-node API 转 tensor

  const feeds = { images: inputTensor }; // 'images' 是模型输入名字
  const results = await session.run(feeds);

  console.log(results);
}

runYOLO();
