#![no_std]

//! # Rigel Timing
//!
//! No-std timing and parameter smoothing infrastructure for audio DSP.
//!
//! This crate provides foundational components for time-based DSP processing:
//!
//! - [`Timebase`] - Sample-accurate global timing context
//! - [`Smoother`] - Parameter smoothing with multiple curve types
//! - [`ControlRateClock`] - Scheduling for control-rate updates
//! - [`ModulationSource`] - Trait interface for modulation sources
//!
//! All types are `Copy`/`Clone`, zero-allocation, and suitable for real-time use.
//!
//! # Example
//!
//! ```ignore
//! use rigel_timing::{Timebase, Smoother, SmoothingMode, ControlRateClock};
//!
//! // Create timing infrastructure
//! let mut timebase = Timebase::new(44100.0);
//! let mut smoother = Smoother::new(1000.0, SmoothingMode::Logarithmic, 10.0, 44100.0);
//! let mut clock = ControlRateClock::new(64);
//!
//! // Process audio block
//! timebase.advance_block(64);
//! for offset in clock.advance(64) {
//!     // Update modulation sources at control rate
//! }
//! ```

mod control_rate;
mod modulation;
mod smoother;
mod timebase;

// Re-export all public types
pub use control_rate::{ControlRateClock, ControlRateUpdates};
pub use modulation::ModulationSource;
pub use smoother::{Smoother, SmoothingMode};
pub use timebase::Timebase;

/// Default smoothing time in milliseconds
pub const DEFAULT_SMOOTHING_TIME_MS: f32 = 5.0;

/// Default sample rate in Hz
pub const DEFAULT_SAMPLE_RATE: f32 = 44100.0;
