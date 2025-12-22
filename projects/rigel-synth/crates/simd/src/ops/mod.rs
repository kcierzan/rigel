//! Functional-style vector operations (T045)
//!
//! This module provides a functional API for SIMD operations as an alternative
//! to the method-based SimdVector trait API. These functions are generic over
//! any type implementing SimdVector.
//!
//! # Design Philosophy
//!
//! The ops module offers:
//! - **Functional style**: Pure functions that take vectors and return vectors
//! - **Composability**: Easy to chain operations in pipelines
//! - **Type inference**: Rust can often infer the SIMD vector type
//! - **Clarity**: Explicit operation names in DSP code
//!
//! # Usage Comparison
//!
//! Method style (via trait):
//! ```rust
//! use rigel_simd::{DefaultSimdVector, SimdVector};
//!
//! let a = DefaultSimdVector::splat(2.0);
//! let b = DefaultSimdVector::splat(3.0);
//! let result = a.add(b).mul(DefaultSimdVector::splat(0.5));
//! ```
//!
//! Functional style (via ops):
//! ```rust
//! use rigel_simd::{DefaultSimdVector, SimdVector};
//! use rigel_simd::ops::{add, mul};
//!
//! let a = DefaultSimdVector::splat(2.0);
//! let b = DefaultSimdVector::splat(3.0);
//! let gain = DefaultSimdVector::splat(0.5);
//! let result = mul(add(a, b), gain);
//! ```
//!
//! # Module Organization
//!
//! - [`arithmetic`]: Basic arithmetic (add, sub, mul, div, neg)
//! - [`fma`]: Fused multiply-add operations
//! - [`minmax`]: Min/max and absolute value
//! - [`compare`]: Comparison operations returning masks
//! - [`horizontal`]: Horizontal reductions (sum, max, min)

pub mod arithmetic;
pub mod compare;
pub mod fma;
pub mod horizontal;
pub mod minmax;

// Re-export all operations for convenience
pub use arithmetic::{add, div, mul, neg, sub};
pub use compare::{eq, ge, gt, le, lt, ne};
pub use fma::{fma, fms, fnma};
pub use horizontal::{horizontal_max, horizontal_min, horizontal_sum};
pub use minmax::{abs, clamp, max, min};
