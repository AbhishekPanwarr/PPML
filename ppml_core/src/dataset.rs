use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CofheItemInput {
    #[serde(rename = "ctHash")]
    pub ct_hash: String,
    #[serde(rename = "securityZone")]
    pub security_zone: u8,
    pub utype: u8,
    pub signature: String,
}

pub fn load_encrypted_dataset(file_path: &str) -> Result<Vec<CofheItemInput>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let dataset: Vec<CofheItemInput> = serde_json::from_reader(reader)?;

    if dataset.is_empty() {
        return Err("encrypted dataset payload is empty".into());
    }

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_json_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "{name}_{}_dataset.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock drift")
                .as_nanos()
        ))
    }

    #[test]
    fn loads_dummy_cofhe_dataset_payload() {
        let path = temp_json_path("cofhe");
        let payload = r#"
[
  {
    "ctHash": "12345678901234567890",
    "securityZone": 0,
    "utype": 4,
    "signature": "0xdeadbeef"
  },
  {
    "ctHash": "98765432109876543210",
    "securityZone": 0,
    "utype": 4,
    "signature": "0xbeadfeed"
  }
]
"#;

        fs::write(&path, payload).expect("dummy dataset written");

        let items = load_encrypted_dataset(path.to_str().expect("utf8 path")).expect("dataset loads");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].ct_hash, "12345678901234567890");
        assert_eq!(items[0].security_zone, 0);
        assert_eq!(items[0].utype, 4);
        assert_eq!(items[0].signature, "0xdeadbeef");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_empty_dataset_payload() {
        let path = temp_json_path("empty");
        fs::write(&path, "[]").expect("empty dataset written");

        let error = load_encrypted_dataset(path.to_str().expect("utf8 path")).expect_err("empty payload rejected");
        assert!(error.to_string().contains("empty"));

        let _ = fs::remove_file(path);
    }
}
