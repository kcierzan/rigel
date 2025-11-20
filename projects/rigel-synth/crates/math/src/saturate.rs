//! Polynomial saturation curves for waveshaping
//!
//! Provides various saturation/clipping functions for harmonic richness in audio DSP.

use crate::traits::SimdVector;

/// Soft clipping using tanh-style saturation
///
/// Provides smooth, symmetric saturation that approaches ±1 asymptotically.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::saturate::soft_clip;
///
/// let x = DefaultSimdVector::splat(2.0);
/// let result = soft_clip(x);
/// // result approaches 1.0 smoothly
/// ```
#[inline(always)]
pub fn soft_clip<V: SimdVector<Scalar = f32>>(x: V) -> V {
    // Use tanh approximation for smooth saturation
    use crate::math::tanh;
    tanh(x)
}

/// Hard clipping (brickwall limiter)
///
/// Clips values to the range [-limit, +limit] with sharp transitions.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::saturate::hard_clip;
///
/// let x = DefaultSimdVector::splat(2.0);
/// let result = hard_clip(x, 1.0);
/// // result = 1.0 (hard clipped)
/// ```
#[inline(always)]
pub fn hard_clip<V: SimdVector<Scalar = f32>>(x: V, limit: f32) -> V {
    let neg_limit = V::splat(-limit);
    let pos_limit = V::splat(limit);
    x.max(neg_limit).min(pos_limit)
}

/// Asymmetric saturation (tube-style)
///
/// Provides different saturation characteristics for positive and negative values,
/// similar to tube amplifier behavior.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::saturate::asymmetric_saturate;
///
/// let x = DefaultSimdVector::splat(1.5);
/// let result = asymmetric_saturate(x, 1.2, 0.8);
/// ```
#[inline(always)]
pub fn asymmetric_saturate<V: SimdVector<Scalar = f32>>(
    x: V,
    pos_threshold: f32,
    neg_threshold: f32,
) -> V {
    let zero = V::splat(0.0);
    let pos_thresh = V::splat(pos_threshold);
    let neg_thresh = V::splat(-neg_threshold);

    // Different limiting for positive and negative
    let is_positive = x.gt(zero);

    let pos_clipped = x.min(pos_thresh);
    let neg_clipped = x.max(neg_thresh);

    V::select(is_positive, pos_clipped, neg_clipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultSimdVector;

    #[test]
    fn test_soft_clip_range() {
        let x = DefaultSimdVector::splat(10.0);
        let result = soft_clip(x);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!(value <= 1.0, "Soft clip should not exceed 1.0");
        assert!(
            value >= 0.99,
            "Soft clip of large value should approach 1.0"
        );
    }

    #[test]
    fn test_hard_clip() {
        let x = DefaultSimdVector::splat(2.0);
        let result = hard_clip(x, 1.0);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!(
            (value - 1.0).abs() < 1e-5,
            "Hard clip should be exactly 1.0"
        );
    }
}
