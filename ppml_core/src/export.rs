use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::model::LogisticModel;
use crate::quantization::config::QuantConfig;
use crate::tensor::{FheTensorOps, TensorError};

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
    pub quantization: QuantizationMetadata,
    pub layers: Vec<LayerMetadata>,
}

#[derive(Debug, Serialize)]
pub struct QuantizationMetadata {
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
            quantization: QuantizationMetadata {
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

    #[test]
    fn writes_frontend_compatible_model_export_json() {
        let quant = QuantConfig::q16f8();
        let quantizer = Quantizer::new(quant.clone());
        let (client_key, ctx) = FheContext::generate_keys_q16f8();
        let model = LogisticModel::zeros(3, quantizer, &client_key, ctx.clone())
            .expect("zero model");

        let temp_path = std::env::temp_dir().join(format!(
            "model_export_{}.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock drift")
                .as_nanos()
        ));

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
}
