import { mkdir, stat } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { pipeline } from "node:stream/promises";
import path from "node:path";

const OUT_DIR = path.resolve("./cicids2017_local_data");

const DATASET_VARIANTS = [
  ["Monday-WorkingHours.pcap_ISCX.csv"],
  ["Tuesday-WorkingHours.pcap_ISCX.csv"],
  ["Wednesday-workingHours.pcap_ISCX.csv"],
  ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"],
  [
    "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
  ],
  ["Friday-WorkingHours-Morning.pcap_ISCX.csv"],
  ["Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"],
  ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
];

const MIRRORS = [
  "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/",
  "https://gitlab.com/msc-cybersecurity-datasets/cicids2017/-/raw/main/",
  "https://raw.githubusercontent.com/cyber-risk-analysis/cicids2017-mirror/main/",
];

async function hasValidFile(filePath) {
  try {
    const info = await stat(filePath);
    return info.size > 0;
  } catch {
    return false;
  }
}

async function downloadFile(url, outPath) {
  const response = await fetch(url, { method: "GET", redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status}`);
  }

  const outStream = createWriteStream(outPath);
  await pipeline(response.body, outStream);
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });

  console.log("=== Descarga CICIDS2017 (mirrors con fallback) ===");
  console.log(`Directorio destino: ${OUT_DIR}`);

  for (const variants of DATASET_VARIANTS) {
    const canonical = variants[0];
    const canonicalPath = path.join(OUT_DIR, canonical);

    if (await hasValidFile(canonicalPath)) {
      console.log(`OK local: ${canonical}`);
      continue;
    }

    let downloaded = false;

    for (const mirror of MIRRORS) {
      for (const variant of variants) {
        const url = `${mirror}${variant}`;
        try {
          console.log(`Intentando ${url}`);
          await downloadFile(url, canonicalPath);
          if (await hasValidFile(canonicalPath)) {
            console.log(`Descargado: ${canonical}`);
            downloaded = true;
            break;
          }
          throw new Error("Archivo vacio");
        } catch (error) {
          console.log(`Fallo ${variant} en mirror: ${error.message}`);
        }
      }
      if (downloaded) {
        break;
      }
    }

    if (!downloaded) {
      throw new Error(`No se pudo descargar ${canonical} desde ningun mirror.`);
    }
  }

  console.log("Descarga completada. Ya puedes ejecutar npx serve .");
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exit(1);
});
