# rigel-simd

Trait-based SIMD abstraction layer for real-time audio DSP.

## Overview

This crate provides zero-cost SIMD abstractions that enable writing DSP algorithms once
and compiling to optimal SIMD instructions for each platform without `#[cfg]` directives.

## Features

- **`SimdVector` trait**: Write backend-agnostic code
- **Compile-time backend selection**: scalar, AVX2, AVX512, or NEON
- **Block processing**: Fixed-size aligned buffers (64/128 samples)
- **Denormal protection**: RAII-based FTZ/DAZ flags
- **No allocations**: Stack-based, real-time safe

## Backend Selection

Select backends via Cargo features:

```toml
[dependencies]
rigel-simd = { path = "...", features = ["avx2"] }
```

Available features:
- `scalar` (default) - 1-lane fallback
- `avx2` - 8-lane x86-64
- `avx512` - 16-lane x86-64 (experimental)
- `neon` - 4-lane ARM64

## Usage

```rust
use rigel_simd::{Block64, DefaultSimdVector, DenormalGuard, SimdVector};
use rigel_simd::ops::mul;

fn apply_gain(input: &Block64, output: &mut Block64, gain: f32) {
    let _guard = DenormalGuard::new();
    let gain_vec = DefaultSimdVector::splat(gain);

    for mut out_chunk in output.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
        let value = out_chunk.load();
        out_chunk.store(mul(value, gain_vec));
    }
}
```

## License

MIT OR Apache-2.0
