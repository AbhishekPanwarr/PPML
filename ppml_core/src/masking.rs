use rand::rngs::OsRng;
use rand::Rng;

use crate::tensor::encrypted::EncryptedTensor;
use crate::tensor::noise::FheOp;
use crate::tensor::FheTensorOps;

pub fn generate_mask(shape: &[usize]) -> Vec<u32> {
    let len = shape.iter().product::<usize>();
    let mut rng = OsRng;
    (0..len).map(|_| rng.gen::<u32>()).collect()
}

pub fn mask_encrypted_tensor(encrypted_tensor: &EncryptedTensor, mask: &[u32]) -> EncryptedTensor {
    assert_eq!(
        encrypted_tensor.data().len(),
        mask.len(),
        "mask length must match encrypted tensor element count"
    );

    let masked_data = encrypted_tensor
        .data()
        .iter()
        .zip(mask.iter())
        .map(|(ciphertext, mask_value)| {
            encrypted_tensor
                .ctx
                .server_key
                .scalar_add_parallelized(ciphertext, u64::from(*mask_value))
        })
        .collect::<Vec<_>>();

    let mut noise = encrypted_tensor.noise().clone();
    noise
        .apply(FheOp::Add)
        .expect("plaintext masking should keep noise metadata valid");

    EncryptedTensor::from_parts(
        masked_data,
        encrypted_tensor.shape().clone(),
        noise,
        encrypted_tensor.ctx.clone(),
    )
    .expect("masking preserves tensor shape")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::FheContext;
    use crate::quantization::config::QuantConfig;
    use crate::quantization::quantizer::Quantizer;
    use crate::tensor::TensorShape;

    #[test]
    fn generates_mask_with_expected_len() {
        let mask = generate_mask(&[2, 3]);
        assert_eq!(mask.len(), 6);
    }

    #[test]
    fn masks_encrypted_tensor_with_plaintext_addition() {
        let quant = QuantConfig::q16f8();
        let quantizer = Quantizer::new(quant.clone());
        let (client_key, ctx) = FheContext::generate_keys_q16f8();
        let tensor = quantizer
            .encrypt_quantized(
                &[1, 2, 3],
                TensorShape::from_2d(1, 3).unwrap(),
                &client_key,
                ctx,
            )
            .expect("encrypt tensor");
        let mask = vec![5, 6, 7];

        let masked = mask_encrypted_tensor(&tensor, &mask);
        let decrypted = masked
            .data()
            .iter()
            .map(|ct| quantizer.decrypt_quantized(ct, &client_key))
            .collect::<Vec<_>>();

        assert_eq!(decrypted, vec![6, 8, 10]);
    }
}
