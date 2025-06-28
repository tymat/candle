# Metal F64 to F32 Conversion Fix

## Problem
Metal backend doesn't support F64 to F32 conversion, causing errors in LayerNorm when epsilon (f64) needs to be converted to F32 tensors.

## Fix Applied
In `candle-nn/src/layer_norm.rs` (line 132), changed from:
```rust
let eps_tensor = Tensor::new(&[self.eps], x.device())?.to_dtype(internal_dtype)?;
```

To:
```rust
let eps_tensor = match internal_dtype {
    DType::F32 => Tensor::new(&[self.eps as f32], x.device())?,
    DType::F64 => Tensor::new(&[self.eps], x.device())?,
    DType::F16 => Tensor::new(&[self.eps as f32], x.device())?.to_dtype(DType::F16)?,
    DType::BF16 => Tensor::new(&[self.eps as f32], x.device())?.to_dtype(DType::BF16)?,
    _ => return Err(candle::Error::UnsupportedDTypeForOp(internal_dtype, "layer_norm").bt()),
};
```

## Additional Fixes Needed
Similar fix needs to be applied to `ops.rs`:
1. In `layer_norm_slow` function (line 897)
2. In `rms_norm_slow` function (for the epsilon addition)

## Testing
To test on Metal:
```bash
cargo test --features metal
```

## Branch
This fix is in branch: `fix/metal-layernorm-f64-f32`