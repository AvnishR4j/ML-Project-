const fs = require("fs"), path = require("path"), eng = require("./engine.js");
const dir = path.join(__dirname, "models"), M = {};
for (const f of fs.readdirSync(dir)) if (f.endsWith(".json")) M[f.replace(".json", "")] = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
const E = eng.Engine(M), texts = JSON.parse(fs.readFileSync(path.join(__dirname, "parity_in.json"), "utf8"));
console.log(JSON.stringify(texts.map((t) => eng.analyze(E, t))));
