// detection-worker.js - Web Worker for ONNX Inference
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

// Configure ONNX Runtime for Web Worker - OPTIMIZED
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.numThreads = 4; // Use multiple threads for faster processing
ort.env.wasm.simd = true; // Enable SIMD for faster computation

let session = null;
const MODEL_INPUT_SIZE = 640;

// Initialize the ONNX session
self.onmessage = async function (e) {
  const { type, data } = e.data;

  try {
    switch (type) {
      case "INIT":
        await initModel(data.modelPath);
        break;

      case "PROCESS_FRAME":
        await processFrame(data);
        break;

      default:
        console.warn("Unknown message type:", type);
    }
  } catch (error) {
    self.postMessage({
      type: "ERROR",
      error: error.message,
    });
  }
};

async function initModel(modelPath) {
  try {
    const modelResponse = await fetch(modelPath);
    const modelBuffer = await modelResponse.arrayBuffer();

    // Try execution providers in order of performance
    // Note: WebGPU doesn't work in workers, but WebGL might
    const sessionOptions = {
      executionProviders: ["webgl", "wasm"],
      graphOptimizationLevel: "all",
      executionMode: "parallel",
      enableCpuMemArena: true,
      enableMemPattern: true,
    };

    session = await ort.InferenceSession.create(modelBuffer, sessionOptions);

    console.log(
      "[Worker] Model loaded with provider:",
      session.handler._backendHint || "unknown"
    );

    self.postMessage({
      type: "MODEL_LOADED",
      inputNames: session.inputNames,
      outputNames: session.outputNames,
    });
  } catch (error) {
    self.postMessage({
      type: "ERROR",
      error: `Failed to load model: ${error.message}`,
    });
  }
}

async function processFrame(data) {
  if (!session) {
    self.postMessage({
      type: "ERROR",
      error: "Model not initialized",
    });
    return;
  }

  const { imageData, frameId, timestamp } = data;
  const startTime = performance.now();

  console.log(`[Worker] Frame ${frameId}: Started processing`);

  try {
    // Preprocess
    const preprocessStart = performance.now();
    const inputTensor = preprocessImageData(imageData);
    const preprocessTime = (performance.now() - preprocessStart).toFixed(2);

    // Inference
    const inferenceStart = performance.now();
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    const results = await session.run(feeds);
    const inferenceTime = (performance.now() - inferenceStart).toFixed(2);

    // Get output
    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];

    const endTime = performance.now();
    const totalTime = (endTime - startTime).toFixed(2);

    console.log(
      `[Worker] Frame ${frameId}: COMPLETE - Total: ${totalTime}ms (Preprocess: ${preprocessTime}ms, Inference: ${inferenceTime}ms)`
    );

    // Send results back to main thread - optimize transfer
    const outputData = new Float32Array(outputTensor.data);

    self.postMessage(
      {
        type: "FRAME_PROCESSED",
        frameId: frameId,
        timestamp: timestamp,
        output: {
          data: outputData,
          dims: outputTensor.dims,
        },
        processingTime: totalTime,
      },
      [outputData.buffer] // Transfer ownership for zero-copy
    );
  } catch (error) {
    const errorTime = (performance.now() - startTime).toFixed(2);
    console.error(
      `[Worker] Frame ${frameId}: ERROR after ${errorTime}ms -`,
      error
    );
    self.postMessage({
      type: "ERROR",
      error: `Processing failed: ${error.message}`,
      frameId: frameId,
    });
  }
}

function preprocessImageData(imageData) {
  const { data, width, height } = imageData;

  // Convert array back to typed array if needed
  const pixelData = Array.isArray(data) ? new Uint8ClampedArray(data) : data;

  //   console.log(
  //     `Worker: Processing image ${width}x${height}, data length: ${pixelData.length}`
  //   );

  const float32Data = new Float32Array(3 * width * height);

  for (let i = 0; i < pixelData.length / 4; i++) {
    const r = pixelData[i * 4] / 255.0;
    const g = pixelData[i * 4 + 1] / 255.0;
    const b = pixelData[i * 4 + 2] / 255.0;
    float32Data[i] = r;
    float32Data[i + width * height] = g;
    float32Data[i + 2 * width * height] = b;
  }

  return new ort.Tensor("float32", float32Data, [1, 3, width, height]);
}
