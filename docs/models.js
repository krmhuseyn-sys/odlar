// Saytdakı modellər. Yeni model əlavə etmək üçün:
//   python -m odlar.export runs/odlar2 --name odlar2 --to docs
// sonra bu siyahıya bir obyekt əlavə et (export komandası blockSize-ı yazır).

export const MODELS = [
  {
    id: "odlar-1.5",
    name: "Odlar-1.5",
    onnx: "odlar15.onnx",
    tok: "tok15.json",
    blockSize: 512,
    size: "49 MB",
    summary: "Daha uzun və əlaqəli mətn yazır",
    about: "Azərbaycan Vikipediyası və FineWeb-2 veb mətnləri (xəbər, bloq, saytlar) üzərində sıfırdan öyrədilib. Odlar-1-dən üç dəfə böyükdür.",
    facts: [["40M", "parametr"], ["345M", "token təlim datası"], ["512", "token kontekst"], ["2 saat", "təlim, bir T4 GPU"]],
  },
  {
    id: "odlar-1",
    name: "Odlar-1",
    onnx: "odlar1.onnx",
    tok: "tok.json",
    blockSize: 256,
    size: "17 MB",
    summary: "İlk model, ensiklopedik üslub",
    about: "Yalnız Azərbaycan Vikipediyası üzərində sıfırdan öyrədilib. Ailənin ilk və ən balaca üzvüdür, tez yüklənir.",
    facts: [["13.8M", "parametr"], ["85M", "token təlim datası"], ["256", "token kontekst"], ["22 dəq", "təlim, bir T4 GPU"]],
  },
];

export const DEFAULT_MODEL = "odlar-1.5";
