use std::error::Error;
use std::fs;
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::dataset::CofheItemInput;
use crate::metadata::{EncryptionMetadata, PreprocessingMetadata, QuantizationMetadata};
use crate::model::LogisticModel;
use crate::quantization::config::QuantConfig;
use crate::tensor::{FheTensorOps, TensorError};

#[derive(Debug, Clone, Serialize)]
pub struct ExportedModel {
    pub model: ExportedModelInfo,
    pub input_schema: InputSchema,
    pub quantization: QuantizationMetadata,
    pub preprocessing: PreprocessingMetadata,
    pub encryption: EncryptionMetadata,
    pub input_schema_hash: String,
    pub context_hash: String,
    pub encrypted_tensors: ExportedEncryptedTensors,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportedModelInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct InputSchema {
    pub rows: usize,
    pub columns: usize,
    pub features: Vec<String>,
    #[serde(rename = "featureTypes")]
    pub feature_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportedEncryptedTensors {
    pub weights: Vec<CofheItemInput>,
    pub bias: Vec<CofheItemInput>,
}

#[derive(Debug, Serialize)]
pub struct ModelExport {
    pub schema_version: u32,
    pub model_type: &'static str,
    pub weights: Vec<i64>,
    pub bias: Vec<i64>,
    pub metadata: ModelMetadata,
    pub encrypted_tensors: EncryptedTensorArtifacts,
}

#[derive(Debug, Serialize)]
pub struct ModelMetadata {
    pub backend: String,
    pub quantization: LegacyQuantizationMetadata,
    pub layers: Vec<LayerMetadata>,
}

#[derive(Debug, Serialize)]
pub struct LegacyQuantizationMetadata {
    pub frac_bits: u32,
    pub total_bits: u32,
    pub scale: i64,
    pub q_min: i64,
    pub q_max: i64,
}

#[derive(Debug, Serialize)]
pub struct LayerMetadata {
    pub name: &'static str,
    pub rows: usize,
    pub cols: usize,
    pub element_count: usize,
    pub encrypted_byte_len: usize,
}

#[derive(Debug, Serialize)]
pub struct EncryptedTensorArtifacts {
    pub weights: EncryptedTensorArtifact,
    pub bias: EncryptedTensorArtifact,
}

#[derive(Debug, Serialize)]
pub struct EncryptedTensorArtifact {
    pub rows: usize,
    pub cols: usize,
    pub bytes: Vec<u8>,
}

pub fn export_to_json(model: &ExportedModel, output_path: &str) -> Result<(), Box<dyn Error>> {
    let json = serde_json::to_vec_pretty(model)?;
    fs::write(output_path, json)?;
    Ok(())
}

pub fn compute_input_schema_hash(input_schema: &InputSchema) -> Result<String, Box<dyn Error>> {
    hash_serialized(input_schema)
}

pub fn compute_context_hash(encryption: &EncryptionMetadata) -> Result<String, Box<dyn Error>> {
    hash_serialized(encryption)
}

pub fn with_compatibility_hashes(mut model: ExportedModel) -> Result<ExportedModel, Box<dyn Error>> {
    model.input_schema_hash = compute_input_schema_hash(&model.input_schema)?;
    model.context_hash = compute_context_hash(&model.encryption)?;
    Ok(model)
}

fn hash_serialized<T: Serialize>(value: &T) -> Result<String, Box<dyn Error>> {
    let bytes = serde_json::to_vec(value)?;
    let digest = Sha256::digest(bytes);
    Ok(format!("0x{:x}", digest))
}

pub fn write_model_export(
    path: &Path,
    model: &LogisticModel,
    quant: &QuantConfig,
    backend_name: &str,
    quantized_weights: Vec<i64>,
    quantized_bias: Vec<i64>,
) -> Result<(), TensorError> {
    let weight_bytes = model.weights.to_bytes()?;
    let bias_bytes = model.bias.to_bytes()?;

    let export = ModelExport {
        schema_version: 1,
        model_type: "encrypted_logistic_regression",
        weights: quantized_weights,
        bias: quantized_bias,
        metadata: ModelMetadata {
            backend: backend_name.to_string(),
            quantization: LegacyQuantizationMetadata {
                frac_bits: quant.frac_bits,
                total_bits: quant.total_bits,
                scale: quant.scale,
                q_min: quant.q_min,
                q_max: quant.q_max,
            },
            layers: vec![
                LayerMetadata {
                    name: "weights",
                    rows: model.weights.shape().rows(),
                    cols: model.weights.shape().cols(),
                    element_count: model.weights.shape().rows() * model.weights.shape().cols(),
                    encrypted_byte_len: weight_bytes.len(),
                },
                LayerMetadata {
                    name: "bias",
                    rows: model.bias.shape().rows(),
                    cols: model.bias.shape().cols(),
                    element_count: model.bias.shape().rows() * model.bias.shape().cols(),
                    encrypted_byte_len: bias_bytes.len(),
                },
            ],
        },
        encrypted_tensors: EncryptedTensorArtifacts {
            weights: EncryptedTensorArtifact {
                rows: model.weights.shape().rows(),
                cols: model.weights.shape().cols(),
                bytes: weight_bytes,
            },
            bias: EncryptedTensorArtifact {
                rows: model.bias.shape().rows(),
                cols: model.bias.shape().cols(),
                bytes: bias_bytes,
            },
        },
    };

    let json = serde_json::to_vec_pretty(&export)
        .map_err(|error| TensorError::Io(format!("while serializing model export: {error}")))?;
    fs::write(path, json)
        .map_err(|error| TensorError::Io(format!("while writing model export: {error}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::context::FheContext;
    use crate::model::LogisticModel;
    use crate::quantization::quantizer::Quantizer;

    fn temp_json_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "{name}_{}_model_export.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock drift")
                .as_nanos()
        ))
    }

    #[test]
    fn writes_frontend_compatible_model_export_json() {
        let quant = QuantConfig::q16f8();
        let quantizer = Quantizer::new(quant.clone());
        let (client_key, ctx) = FheContext::generate_keys_q16f8();
        let model = LogisticModel::zeros(3, quantizer, &client_key, ctx.clone())
            .expect("zero model");

        let temp_path = temp_json_path("legacy");

        write_model_export(
            &temp_path,
            &model,
            &quant,
            ctx.backend_name(),
            vec![1, 2, 3],
            vec![0],
        )
        .expect("export succeeds");

        let json = fs::read_to_string(&temp_path).expect("export file readable");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");

        assert_eq!(parsed["weights"], serde_json::json!([1, 2, 3]));
        assert_eq!(parsed["bias"], serde_json::json!([0]));
        assert_eq!(parsed["metadata"]["quantization"]["scale"], serde_json::json!(8));
        assert_eq!(parsed["metadata"]["layers"][0]["rows"], serde_json::json!(3));
        assert!(parsed["encrypted_tensors"]["weights"]["bytes"]
            .as_array()
            .expect("byte array")
            .len()
            > 0);

        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn exported_model_serializes_cthash_as_string_and_includes_hashes() {
        let temp_path = temp_json_path("cofhe");
        let model = with_compatibility_hashes(ExportedModel {
            model: ExportedModelInfo {
                name: "Diabetes-LogReg".to_string(),
                model_type: "logistic_regression".to_string(),
            },
            input_schema: InputSchema {
                rows: 1,
                columns: 3,
                features: vec!["Glucose".to_string(), "BMI".to_string(), "Age".to_string()],
                feature_types: vec!["u32".to_string(), "u32".to_string(), "u32".to_string()],
            },
            quantization: QuantizationMetadata {
                scale: 1000,
                scheme: "fixed_point_u32".to_string(),
            },
            preprocessing: PreprocessingMetadata {
                normalization: "none".to_string(),
                feature_order_locked: true,
            },
            encryption: EncryptionMetadata {
                scheme: "fhenix_cofhe".to_string(),
                chain_id: 11155111,
                security_zone: 0,
                context_hash: "0xabc123".to_string(),
            },
            input_schema_hash: String::new(),
            context_hash: String::new(),
            encrypted_tensors: ExportedEncryptedTensors {
                weights: vec![
                    CofheItemInput {
                        ct_hash: "12345678901234567890".to_string(),
                        security_zone: 0,
                        utype: 4,
                        signature: "0xdeadbeef".to_string(),
                    },
                    CofheItemInput {
                        ct_hash: "22345678901234567890".to_string(),
                        security_zone: 0,
                        utype: 4,
                        signature: "0xfeedbead".to_string(),
                    },
                    CofheItemInput {
                        ct_hash: "32345678901234567890".to_string(),
                        security_zone: 0,
                        utype: 4,
                        signature: "0xcafebabe".to_string(),
                    },
                ],
                bias: vec![CofheItemInput {
                    ct_hash: "42345678901234567890".to_string(),
                    security_zone: 0,
                    utype: 4,
                    signature: "0xfacefeed".to_string(),
                }],
            },
        })
        .expect("hashes computed");

        export_to_json(&model, temp_path.to_str().expect("utf8 path")).expect("json exported");

        let json = fs::read_to_string(&temp_path).expect("export file readable");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");

        assert_eq!(
            parsed["encrypted_tensors"]["weights"][0]["ctHash"],
            serde_json::Value::String("12345678901234567890".to_string())
        );
        assert!(parsed["input_schema_hash"]
            .as_str()
            .expect("schema hash string")
            .starts_with("0x"));
        assert!(parsed["context_hash"]
            .as_str()
            .expect("context hash string")
            .starts_with("0x"));

        let _ = fs::remove_file(temp_path);
    }
}
