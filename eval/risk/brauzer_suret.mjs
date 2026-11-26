import fs from "fs";
import * as ort from "onnxruntime-web";
ort.env.wasm.numThreads = 1;
for (const f of ["./odlar2_test.onnx", "../../docs/odlar15.onnx"]) {
  const s = await ort.InferenceSession.create(fs.readFileSync(f), { executionProviders: ["wasm"] });
  const V = f.includes("odlar2") ? 32000 : 16000;
  for (const n of [32, 256, 512, 1024]) {
    if (!f.includes("odlar2") && n > 512) continue;
    const ids = BigInt64Array.from({ length: n }, (_, i) => BigInt((i * 7) % V));
    const t = new ort.Tensor("int64", ids, [1, n]);
    await s.run({ input_ids: t });
    const t0 = performance.now(); const R = 3;
    for (let r = 0; r < R; r++) await s.run({ input_ids: t });
    const ms = (performance.now() - t0) / R;
    console.log(f.split("/").pop(), "kontekst", n, ":", ms.toFixed(0), "ms/token =>", (1000 / ms).toFixed(1), "token/san");
  }
}
