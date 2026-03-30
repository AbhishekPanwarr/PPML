use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetMetadata {
    pub rows: usize,
    pub columns: usize,
    pub features: Vec<String>,
    #[serde(rename = "featureTypes")]
    pub feature_types: Vec<String>,
    pub quantization: QuantizationMetadata,
    pub preprocessing: PreprocessingMetadata,
    pub encryption: EncryptionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuantizationMetadata {
    pub scale: u32,
    pub scheme: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreprocessingMetadata {
    pub normalization: String,
    #[serde(rename = "featureOrderLocked")]
    pub feature_order_locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncryptionMetadata {
    pub scheme: String,
    #[serde(rename = "chainId")]
    pub chain_id: u64,
    #[serde(rename = "securityZone")]
    pub security_zone: u8,
    #[serde(rename = "contextHash")]
    pub context_hash: String,
}

pub fn load_dataset_metadata(file_path: &str) -> Result<DatasetMetadata, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let metadata: DatasetMetadata = serde_json::from_reader(reader)?;

    if metadata.rows == 0 {
        return Err("dataset metadata rows must be greater than zero".into());
    }

    if metadata.columns == 0 {
        return Err("dataset metadata columns must be greater than zero".into());
    }

    if metadata.features.is_empty() {
        return Err("dataset metadata features must not be empty".into());
    }

    if metadata.features.len() != metadata.columns {
        return Err("dataset metadata features length must match columns".into());
    }

    if metadata.feature_types.len() != metadata.columns {
        return Err("dataset metadata feature_types length must match columns".into());
    }

    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_json_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "{name}_{}_metadata.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock drift")
                .as_nanos()
        ))
    }

    #[test]
    fn loads_dataset_metadata() {
        let path = temp_json_path("cofhe");
        let payload = r#"
{
  "rows": 2,
  "columns": 3,
  "features": ["Glucose", "BMI", "Age"],
  "featureTypes": ["u32", "u32", "u32"],
  "quantization": {
    "scale": 1000,
    "scheme": "fixed_point_u32"
  },
  "preprocessing": {
    "normalization": "none",
    "featureOrderLocked": true
  },
  "encryption": {
    "scheme": "fhenix_cofhe",
    "chainId": 11155111,
    "securityZone": 0,
    "contextHash": "0xabc123"
  }
}
"#;

        fs::write(&path, payload).expect("metadata written");

        let metadata =
            load_dataset_metadata(path.to_str().expect("utf8 path")).expect("metadata loads");
        assert_eq!(metadata.rows, 2);
        assert_eq!(metadata.columns, 3);
        assert_eq!(metadata.features[0], "Glucose");
        assert_eq!(metadata.quantization.scale, 1000);
        assert_eq!(metadata.encryption.chain_id, 11155111);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_mismatched_feature_shape() {
        let path = temp_json_path("invalid");
        let payload = r#"
{
  "rows": 2,
  "columns": 3,
  "features": ["Glucose", "BMI"],
  "featureTypes": ["u32", "u32", "u32"],
  "quantization": {
    "scale": 1000,
    "scheme": "fixed_point_u32"
  },
  "preprocessing": {
    "normalization": "none",
    "featureOrderLocked": true
  },
  "encryption": {
    "scheme": "fhenix_cofhe",
    "chainId": 11155111,
    "securityZone": 0,
    "contextHash": "0xabc123"
  }
}
"#;

        fs::write(&path, payload).expect("invalid metadata written");

        let error =
            load_dataset_metadata(path.to_str().expect("utf8 path")).expect_err("shape mismatch rejected");
        assert!(error.to_string().contains("features length"));

        let _ = fs::remove_file(path);
    }
}
