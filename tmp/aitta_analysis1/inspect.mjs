import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/anisrahm/Downloads/Aitta performance benchmark analysis (1).xlsx";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 4000,
});
console.log("SHEETS");
console.log(sheets.ndjson);

const sheet = workbook.worksheets.getItemAt(0);
const core = sheet.getRange("A1:AB27");
const target = sheet.getRange("W1:AB27");
const buBlock = sheet.getRange("P1:V27");

console.log("TARGET_VALUES");
console.log(JSON.stringify(target.values));
console.log("TARGET_FORMULAS");
console.log(JSON.stringify(target.formulas));
console.log("BU_BLOCK_VALUES");
console.log(JSON.stringify(buBlock.values));
console.log("BU_BLOCK_FORMULAS");
console.log(JSON.stringify(buBlock.formulas));

console.log("ROWS");
const values = core.values;
const formulas = core.formulas;
for (let r = 4; r < values.length; r += 1) {
  console.log(JSON.stringify({
    row: r + 1,
    model: values[r]?.[0] ?? null,
    gcds: values[r]?.[1] ?? null,
    thetaMaxIn: values[r]?.[2] ?? null,
    thetaMaxOut: values[r]?.[3] ?? null,
    thetaSingleIn: values[r]?.[4] ?? null,
    thetaSingleOut: values[r]?.[5] ?? null,
    compareValues: values[r]?.slice(22, 28) ?? [],
    compareFormulas: formulas[r]?.slice(22, 28) ?? [],
  }));
}

const rows = [];
for (let r = 4; r < values.length; r += 1) {
  const model = values[r]?.[0];
  const [inputSaving, outputSaving, inputImprovement, outputImprovement, combinedSaving, combinedImprovement] = values[r]?.slice(22, 28) ?? [];
  if (typeof combinedImprovement === "number") {
    rows.push({
      row: r + 1,
      model,
      inputSaving,
      outputSaving,
      inputImprovement,
      outputImprovement,
      combinedSaving,
      combinedImprovement,
      outputShare: outputSaving / combinedSaving,
      inputSpeedup: values[r][2] / values[r][4],
      outputSpeedup: values[r][3] / values[r][5],
    });
  }
}

const summarize = (key) => {
  const nums = rows.map((row) => row[key]).sort((a, b) => a - b);
  const sum = nums.reduce((acc, value) => acc + value, 0);
  const mid = Math.floor(nums.length / 2);
  return {
    count: nums.length,
    min: nums[0],
    median: nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2,
    mean: sum / nums.length,
    max: nums[nums.length - 1],
  };
};

console.log("STATS");
console.log(JSON.stringify({
  validModels: rows.length,
  inputImprovement: summarize("inputImprovement"),
  outputImprovement: summarize("outputImprovement"),
  combinedImprovement: summarize("combinedImprovement"),
  combinedSaving: summarize("combinedSaving"),
  outputShare: summarize("outputShare"),
  inputSpeedup: summarize("inputSpeedup"),
  outputSpeedup: summarize("outputSpeedup"),
  zeroInputImprovement: rows.filter((row) => Math.abs(row.inputImprovement) < 1e-12).map((row) => ({row: row.row, model: row.model})),
  topCombined: [...rows].sort((a, b) => b.combinedImprovement - a.combinedImprovement).slice(0, 5),
  bottomCombined: [...rows].sort((a, b) => a.combinedImprovement - b.combinedImprovement).slice(0, 5),
}, null, 2));

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "formula error scan",
  maxChars: 8000,
});
console.log("ERRORS");
console.log(formulaErrors.ndjson);

await fs.mkdir("/Users/anisrahm/Documents/data-aware-ai/tmp/aitta_analysis1", { recursive: true });
const preview = await workbook.render({
  sheetName: sheet.name,
  range: "A1:AB27",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  "/Users/anisrahm/Documents/data-aware-ai/tmp/aitta_analysis1/preview.png",
  new Uint8Array(await preview.arrayBuffer()),
);

const focused = await workbook.render({
  sheetName: sheet.name,
  range: "W1:AB27",
  scale: 2,
  format: "png",
});
await fs.writeFile(
  "/Users/anisrahm/Documents/data-aware-ai/tmp/aitta_analysis1/preview-W-AB.png",
  new Uint8Array(await focused.arrayBuffer()),
);
