use std::fs;
use std::path::{Path, PathBuf};

use ppml_core::context::FheContext;
use ppml_core::export::{
    read_masked_model_package, write_masked_model_package, MaskedModelMetadata,
    MaskedModelPackage,
};
use ppml_core::masking::{generate_mask, mask_encrypted_tensor};
use ppml_core::tensor::{EncryptedTensor, FheTensorOps, TensorError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HospitalReturnPayload {
    session_id: String,
    masked_plain_weights: Vec<u32>,
}

fn main() -> Result<(), TensorError> {
    let input_path = ppml_root().join("hospital_payload.json");
    let output_path = ppml_root().join("hospital_returned_plain.json");
    let key_cache_path = ppml_root().join("fhe_keys.bin");
    let weights_path = ppml_root().join("fhenix_weights.bin");

    let (client_key, ctx) = match FheContext::load_or_generate(&key_cache_path) {
        Ok(keys) => keys,
        Err(error) => {
            eprintln!(
                "warning: failed to load cached hospital key from {}: {error}; generating a mock keypair instead",
                key_cache_path.display()
            );
            FheContext::generate_keys_q16f8()
        }
    };

    if !input_path.exists() {
        let weights_bytes = fs::read(&weights_path).map_err(|error| {
            TensorError::Io(format!(
                "failed to read existing encrypted weights from {}: {error}",
                weights_path.display()
            ))
        })?;
        let encrypted_weights = EncryptedTensor::from_bytes(&weights_bytes, ctx.clone())?;
        let mask_weights = generate_mask(encrypted_weights.shape().dims());
        let masked_encrypted_weights = mask_encrypted_tensor(&encrypted_weights, &mask_weights);
        let bootstrap_package = MaskedModelPackage {
            session_id: format!(
                "verify-existing-{}",
                std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos()
            ),
            masked_encrypted_weights,
            metadata: MaskedModelMetadata {
                backend: ctx.backend_name().to_string(),
                rows: encrypted_weights.shape().rows(),
                cols: encrypted_weights.shape().cols(),
                element_count: encrypted_weights.shape().elem_count(),
            },
        };

        write_masked_model_package(&input_path, &bootstrap_package)?;
        println!(
            "Bootstrapped {} from existing trained weights at {}",
            input_path.display(),
            weights_path.display()
        );
    }

    let package = read_masked_model_package(&input_path, ctx)?;
    let masked_plain_weights = package
        .masked_encrypted_weights
        .data()
        .iter()
        .map(|ciphertext| {
            let raw: u64 = client_key.decrypt(ciphertext);
            u32::try_from(raw).map_err(|_| {
                TensorError::Io(format!(
                    "decrypted masked weight {raw} does not fit into u32 for session {}",
                    package.session_id
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let response = HospitalReturnPayload {
        session_id: package.session_id,
        masked_plain_weights,
    };

    let json = serde_json::to_vec_pretty(&response).map_err(|error| {
        TensorError::Io(format!("while serializing hospital return payload: {error}"))
    })?;
    fs::write(&output_path, json).map_err(|error| {
        TensorError::Io(format!(
            "while writing hospital return payload to {}: {error}",
            output_path.display()
        ))
    })?;

    println!(
        "Hospital decrypted masked weights and wrote {}",
        output_path.display()
    );

    Ok(())
}

fn ppml_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ppml_train lives under the PPML workspace root")
        .to_path_buf()
}
