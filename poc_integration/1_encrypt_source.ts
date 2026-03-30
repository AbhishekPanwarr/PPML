import fs from 'fs';
import { createRequire } from 'module';
import path from 'path';

const workspaceRequire = createRequire(path.resolve(process.cwd(), 'package.json'));
const hre = workspaceRequire('hardhat') as typeof import('hardhat');
const { Encryptable } = workspaceRequire('@cofhe/sdk') as typeof import('@cofhe/sdk');

type DatasetMetadata = {
  rows: number;
  columns: number;
  features: string[];
  featureTypes: string[];
  quantization: {
    scale: number;
    scheme: string;
  };
  preprocessing: {
    normalization: string;
    featureOrderLocked: boolean;
  };
  encryption: {
    scheme: string;
    chainId: number;
    securityZone: number;
    contextHash: string;
  };
};

function bigintReplacer(key: string, value: unknown) {
  if (key === 'ctHash' && typeof value === 'bigint') {
    return value.toString();
  }

  if (typeof value === 'bigint') {
    return value.toString();
  }

  return value;
}

async function main() {
  const outputDir = path.resolve(__dirname);
  const encryptedDatasetPath = path.join(outputDir, 'encrypted_dataset.json');
  const datasetMetadataPath = path.join(outputDir, 'dataset_metadata.json');

  await hre.run('task:cofhe-mocks:deploy', { deployTestBed: false, silent: true });

  const [signer] = await hre.ethers.getSigners();
  const client = await hre.cofhe.createClientWithBatteries(signer);

  const rawData = [10, 20, 30, 40];
  const encryptables = rawData.map((value) => Encryptable.uint32(BigInt(value)));
  const encryptedDataset = await client.encryptInputs(encryptables).execute();

  const metadata: DatasetMetadata = {
    rows: 2,
    columns: 2,
    features: ['FeatureA', 'FeatureB'],
    featureTypes: ['u32', 'u32'],
    quantization: {
      scale: 1000,
      scheme: 'fixed_point_u32',
    },
    preprocessing: {
      normalization: 'none',
      featureOrderLocked: true,
    },
    encryption: {
      scheme: 'fhenix_cofhe',
      chainId: Number((await hre.ethers.provider.getNetwork()).chainId),
      securityZone: 0,
      contextHash: 'poc-localcofhe-context',
    },
  };

  fs.writeFileSync(
    encryptedDatasetPath,
    `${JSON.stringify(encryptedDataset, bigintReplacer, 2)}\n`,
    'utf8',
  );
  fs.writeFileSync(
    datasetMetadataPath,
    `${JSON.stringify(metadata, null, 2)}\n`,
    'utf8',
  );

  console.log(`Encrypted dataset written to ${encryptedDatasetPath}`);
  console.log(`Dataset metadata written to ${datasetMetadataPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
