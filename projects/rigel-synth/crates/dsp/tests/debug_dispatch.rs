use rigel_math::simd::dispatcher::{BackendDispatcher, CpuFeatures};

#[test]
fn debug_runtime_dispatch() {
    let features = CpuFeatures::detect();
    println!("\n=== CPU Features Detected ===");
    println!("  AVX2: {}", features.has_avx2);
    println!("  AVX-512 F: {}", features.has_avx512_f);
    println!("  AVX-512 BW: {}", features.has_avx512_bw);
    println!("  AVX-512 DQ: {}", features.has_avx512_dq);
    println!("  AVX-512 VL: {}", features.has_avx512_vl);
    println!("  AVX-512 Full: {}", features.has_avx512_full());

    let dispatcher = BackendDispatcher::init();
    println!("\n=== Backend Selection ===");
    println!("  Selected backend: {}", dispatcher.backend_name());
    println!("  Backend type: {:?}", dispatcher.backend_type());
    println!("============================\n");
}
