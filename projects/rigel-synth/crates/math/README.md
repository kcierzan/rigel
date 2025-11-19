# rigel-math

Trait-based SIMD abstraction library for real-time audio DSP.

## Features

- **Zero-cost SIMD abstractions**: Write once, compile to optimal SIMD for each platform
- **Backend support**: Scalar, AVX2, AVX512, NEON
- **Real-time safe**: No allocations, deterministic execution
- **Comprehensive**: Block processing, fast math, lookup tables, denormal protection

## Quick Start

```rust
use rigel_math::{Block64, DefaultSimdVector};
use rigel_math::ops::mul;

fn apply_gain(input: &Block64, output: &mut Block64, gain: f32) {
    let gain_vec = DefaultSimdVector::splat(gain);

    for (in_chunk, out_chunk) in input.as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        *out_chunk = mul(*in_chunk, gain_vec);
    }
}
```

## Backend Selection

Choose backend via cargo features (mutually exclusive):

```bash
cargo build --features scalar   # Default, always available
cargo build --features avx2     # x86-64 with AVX2
cargo build --features avx512   # x86-64 with AVX-512
cargo build --features neon     # ARM64 with NEON
```

## Documentation

See the [quickstart guide](../../specs/001-fast-dsp-math/quickstart.md) for more examples.

## License

MIT OR Apache-2.0
