use candle_core::{Device, DType, Module, Result, Tensor};
use candle_nn::{VarBuilder, VarMap, LayerNormConfig};

fn main() -> Result<()> {
    // Try to use Metal device
    let device = match Device::new_metal(0) {
        Ok(device) => {
            println!("Using Metal device");
            device
        }
        Err(_) => {
            println!("Metal not available, using CPU");
            Device::Cpu
        }
    };
    
    // Create a simple LayerNorm test
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let config = LayerNormConfig {
        eps: 1e-5,  // This is f64
        remove_mean: true,
        affine: true,
    };
    
    println!("Creating LayerNorm with eps={}", config.eps);
    let layer_norm = candle_nn::layer_norm(64, config, vb)?;
    
    // Create F32 input
    let input = Tensor::randn(0f32, 1f32, (2, 8, 64), &device)?;
    println!("Input shape: {:?}, dtype: {:?}", input.shape(), input.dtype());
    
    // Forward pass - this is where the error would occur
    println!("Running forward pass...");
    let output = layer_norm.forward(&input)?;
    println!("Output shape: {:?}, dtype: {:?}", output.shape(), output.dtype());
    
    println!("\n✅ Success! LayerNorm works on {:?} without F64->F32 conversion error", device);
    
    Ok(())
}