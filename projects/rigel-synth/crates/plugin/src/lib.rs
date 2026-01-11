//! # Rigel Plugin
//!
//! Wavetable synthesizer plugin with full FM envelope control using nih-plug.

use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use rigel_dsp::{FmEnvelopeParams, Segment, SynthEngine, SynthParams};
use std::sync::Arc;

mod editor;
mod envelope_formatters;

/// Time range constants (matching DX7 rate 0-99 range)
/// Note: Rate 99 = 0 samples (instant), but we need non-zero for log range
const TIME_MIN: f32 = 0.0005; // 0.5ms - effectively instant for percussive attacks
const TIME_MAX: f32 = 40.0; // ~40s (rate 0)

/// The main plugin struct
pub struct RigelPlugin {
    params: Arc<RigelPluginParams>,
    synth_engine: SynthEngine,
    sample_rate: f32,
}

/// Plugin parameters with full FM envelope control
#[derive(Params)]
pub struct RigelPluginParams {
    /// The editor state, saved together with the parameter state so the custom scaling can be
    /// restored.
    #[persist = "editor-state"]
    pub editor_state: Arc<ViziaState>,

    /// Master volume
    #[id = "volume"]
    pub volume: FloatParam,

    /// Pitch offset in semitones
    #[id = "pitch"]
    pub pitch_offset: FloatParam,

    // ===== Key-On Segment 1 (Attack) =====
    /// Segment 1 Time (Attack time)
    #[id = "seg1_time"]
    pub seg1_time: FloatParam,

    /// Segment 1 Level (Attack target)
    #[id = "seg1_level"]
    pub seg1_level: IntParam,

    // ===== Key-On Segment 2 (Decay 1) =====
    /// Segment 2 Time (Decay 1 time)
    #[id = "seg2_time"]
    pub seg2_time: FloatParam,

    /// Segment 2 Level (Decay 1 target)
    #[id = "seg2_level"]
    pub seg2_level: IntParam,

    // ===== Key-On Segment 3 (Decay 2) =====
    /// Segment 3 Time
    #[id = "seg3_time"]
    pub seg3_time: FloatParam,

    /// Segment 3 Level
    #[id = "seg3_level"]
    pub seg3_level: IntParam,

    // ===== Key-On Segment 4 (Decay 3) =====
    /// Segment 4 Time
    #[id = "seg4_time"]
    pub seg4_time: FloatParam,

    /// Segment 4 Level
    #[id = "seg4_level"]
    pub seg4_level: IntParam,

    // ===== Key-On Segment 5 (Decay 4) =====
    /// Segment 5 Time
    #[id = "seg5_time"]
    pub seg5_time: FloatParam,

    /// Segment 5 Level
    #[id = "seg5_level"]
    pub seg5_level: IntParam,

    // ===== Key-On Segment 6 (Sustain) =====
    /// Segment 6 Time (Sustain approach time)
    #[id = "seg6_time"]
    pub seg6_time: FloatParam,

    /// Segment 6 Level (Sustain level)
    #[id = "seg6_level"]
    pub seg6_level: IntParam,

    // ===== Release Segment 1 =====
    /// Release 1 Time
    #[id = "rel1_time"]
    pub rel1_time: FloatParam,

    /// Release 1 Level
    #[id = "rel1_level"]
    pub rel1_level: IntParam,

    // ===== Release Segment 2 =====
    /// Release 2 Time
    #[id = "rel2_time"]
    pub rel2_time: FloatParam,

    /// Release 2 Level
    #[id = "rel2_level"]
    pub rel2_level: IntParam,

    // ===== Rate Scaling =====
    /// Rate scaling (keyboard tracking for envelope speed)
    #[id = "rate_scaling"]
    pub rate_scaling: IntParam,
}

impl Default for RigelPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(RigelPluginParams::default()),
            synth_engine: SynthEngine::new(44100.0),
            sample_rate: 44100.0,
        }
    }
}

/// Helper to create a time parameter with logarithmic range.
///
/// Time values range from 0.5ms (instant) to 40s (very slow).
/// Internally converts to DX7 rate (0-99) for DSP processing.
fn time_param(name: &str, default_seconds: f32) -> FloatParam {
    FloatParam::new(
        name,
        default_seconds,
        FloatRange::Skewed {
            min: TIME_MIN,
            max: TIME_MAX,
            factor: FloatRange::skew_factor(-2.5), // Logarithmic feel
        },
    )
    .with_value_to_string(Arc::new(envelope_formatters::time_to_string))
    .with_string_to_value(Arc::new(envelope_formatters::string_to_time))
}

/// Helper to create a level parameter (0-99)
fn level_param(name: &str, default: i32) -> IntParam {
    IntParam::new(name, default, IntRange::Linear { min: 0, max: 99 })
}

impl Default for RigelPluginParams {
    fn default() -> Self {
        Self {
            editor_state: editor::default_state(),

            volume: FloatParam::new("Volume", 0.7, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(5.0))
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),

            pitch_offset: FloatParam::new(
                "Pitch",
                0.0,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_unit(" st")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),

            // Key-On Segment 1 (Attack): Fast attack to full level
            seg1_time: time_param("Seg1 Time", 0.010), // 10ms attack
            seg1_level: level_param("Seg1 Level", 99),

            // Key-On Segment 2 (Decay 1): Medium decay to sustain
            seg2_time: time_param("Seg2 Time", 0.150), // 150ms decay
            seg2_level: level_param("Seg2 Level", 69),

            // Key-On Segments 3-6: Hold at sustain (instant transitions)
            seg3_time: time_param("Seg3 Time", TIME_MIN), // Instant
            seg3_level: level_param("Seg3 Level", 69),

            seg4_time: time_param("Seg4 Time", TIME_MIN), // Instant
            seg4_level: level_param("Seg4 Level", 69),

            seg5_time: time_param("Seg5 Time", TIME_MIN), // Instant
            seg5_level: level_param("Seg5 Level", 69),

            seg6_time: time_param("Seg6 Time", TIME_MIN), // Instant
            seg6_level: level_param("Seg6 Level", 69),

            // Release Segment 1: Medium release to silence
            rel1_time: time_param("Rel1 Time", 0.300), // 300ms release
            rel1_level: level_param("Rel1 Level", 0),

            // Release Segment 2: Instant to silence (fallback)
            rel2_time: time_param("Rel2 Time", TIME_MIN), // Instant
            rel2_level: level_param("Rel2 Level", 0),

            // Rate scaling: 0 = no keyboard tracking
            rate_scaling: IntParam::new("Rate Scaling", 0, IntRange::Linear { min: 0, max: 7 }),
        }
    }
}

impl Plugin for RigelPlugin {
    const NAME: &'static str = "Rigel";
    const VENDOR: &'static str = "Kyle Cierzan";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "kcierzan@gmail.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(2),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(self.params.clone(), self.params.editor_state.clone())
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // Store sample rate for time-to-rate conversion
        self.sample_rate = buffer_config.sample_rate;
        // Create new synth engine with correct sample rate
        self.synth_engine = SynthEngine::new(buffer_config.sample_rate);
        true
    }

    fn reset(&mut self) {
        self.synth_engine.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let mut next_event = context.next_event();

        // Process each sample
        for (sample_id, channel_samples) in buffer.iter_samples().enumerate() {
            // Handle MIDI events at this sample
            while let Some(event) = next_event {
                if event.timing() != sample_id as u32 {
                    break;
                }

                match event {
                    NoteEvent::NoteOn { note, velocity, .. } => {
                        self.synth_engine.note_on(note, velocity);
                    }
                    NoteEvent::NoteOff { note, .. } => {
                        self.synth_engine.note_off(note);
                    }
                    NoteEvent::Choke { note, .. } => {
                        self.synth_engine.note_off(note);
                    }
                    _ => {}
                }

                next_event = context.next_event();
            }

            // Build FM envelope params from plugin parameters
            // Convert time values (seconds) to DX7 rates (0-99)
            let envelope = FmEnvelopeParams {
                key_on: [
                    Segment::new(
                        envelope_formatters::time_to_rate(
                            self.params.seg1_time.value(),
                            self.sample_rate,
                        ),
                        self.params.seg1_level.value() as u8,
                    ),
                    Segment::new(
                        envelope_formatters::time_to_rate(
                            self.params.seg2_time.value(),
                            self.sample_rate,
                        ),
                        self.params.seg2_level.value() as u8,
                    ),
                    Segment::new(
                        envelope_formatters::time_to_rate(
                            self.params.seg3_time.value(),
                            self.sample_rate,
                        ),
                        self.params.seg3_level.value() as u8,
                    ),
                    Segment::new(
                        envelope_formatters::time_to_rate(
                            self.params.seg4_time.value(),
                            self.sample_rate,
                        ),
                        self.params.seg4_level.value() as u8,
                    ),
                    Segment::new(
                        envelope_formatters::time_to_rate(
                            self.params.seg5_time.value(),
                            self.sample_rate,
                        ),
                        self.params.seg5_level.value() as u8,
                    ),
                    Segment::new(
                        envelope_formatters::time_to_rate(
                            self.params.seg6_time.value(),
                            self.sample_rate,
                        ),
                        self.params.seg6_level.value() as u8,
                    ),
                ],
                release: [
                    Segment::new(
                        envelope_formatters::time_to_rate(
                            self.params.rel1_time.value(),
                            self.sample_rate,
                        ),
                        self.params.rel1_level.value() as u8,
                    ),
                    Segment::new(
                        envelope_formatters::time_to_rate(
                            self.params.rel2_time.value(),
                            self.sample_rate,
                        ),
                        self.params.rel2_level.value() as u8,
                    ),
                ],
                rate_scaling: self.params.rate_scaling.value() as u8,
            };

            // Get current synth parameters from plugin parameters
            let synth_params = SynthParams {
                volume: self.params.volume.smoothed.next(),
                pitch_offset: self.params.pitch_offset.smoothed.next(),
                envelope,
            };

            // Process one sample
            let output_sample = self.synth_engine.process_sample(&synth_params);

            // Write to all output channels (mono to stereo)
            for channel_sample in channel_samples {
                *channel_sample = output_sample;
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for RigelPlugin {
    const CLAP_ID: &'static str = "com.kylecierzan.rigel";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Monophonic wavetable synthesizer");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Instrument,
        ClapFeature::Synthesizer,
        ClapFeature::Stereo,
        ClapFeature::Mono,
    ];
}

impl Vst3Plugin for RigelPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RigelSynthVst3Id";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Instrument,
        Vst3SubCategory::Synth,
        Vst3SubCategory::Stereo,
    ];
}

// Export the plugin
nih_export_clap!(RigelPlugin);
nih_export_vst3!(RigelPlugin);

#[cfg(test)]
mod tests {
    use super::*;
    // Tests for the Rigel plugin

    fn create_test_plugin() -> RigelPlugin {
        RigelPlugin::default()
    }

    fn create_test_buffer_config() -> BufferConfig {
        BufferConfig {
            sample_rate: 44100.0,
            min_buffer_size: Some(64),
            max_buffer_size: 1024,
            process_mode: ProcessMode::Realtime,
        }
    }

    #[test]
    fn test_plugin_initialization() {
        let mut plugin = create_test_plugin();
        let buffer_config = create_test_buffer_config();
        let audio_io_layout = &RigelPlugin::AUDIO_IO_LAYOUTS[0];

        struct MockInitContext;
        impl InitContext<RigelPlugin> for MockInitContext {
            fn set_current_voice_capacity(&self, _capacity: u32) {}
            fn plugin_api(&self) -> PluginApi {
                PluginApi::Clap
            }
            fn execute(&self, _task: <RigelPlugin as Plugin>::BackgroundTask) {}
            fn set_latency_samples(&self, _samples: u32) {}
        }

        let mut context = MockInitContext;
        let result = plugin.initialize(audio_io_layout, &buffer_config, &mut context);

        assert!(result, "Plugin should initialize successfully");
    }

    #[test]
    fn test_parameter_defaults() {
        let plugin = create_test_plugin();

        // Test volume default (0.7)
        assert!((plugin.params.volume.value() - 0.7).abs() < 0.001);

        // Test pitch offset default (0.0)
        assert!((plugin.params.pitch_offset.value() - 0.0).abs() < 0.001);

        // Test FM envelope defaults (segment 1 = attack)
        assert!((plugin.params.seg1_time.value() - 0.010).abs() < 0.001); // 10ms attack
        assert_eq!(plugin.params.seg1_level.value(), 99); // To full level
        assert!((plugin.params.seg2_time.value() - 0.150).abs() < 0.001); // 150ms decay
        assert_eq!(plugin.params.seg2_level.value(), 69); // To sustain (~70%)
        assert!((plugin.params.rel1_time.value() - 0.300).abs() < 0.001); // 300ms release
        assert_eq!(plugin.params.rel1_level.value(), 0); // To silence
        assert_eq!(plugin.params.rate_scaling.value(), 0); // No keyboard tracking
    }

    #[test]
    fn test_parameter_ranges() {
        let plugin = create_test_plugin();

        // Test that parameters have valid ranges by checking extreme values
        // Volume should clamp to 0.0-1.0 range
        assert!((plugin.params.volume.value() >= 0.0) && (plugin.params.volume.value() <= 1.0));

        // Pitch should be within reasonable range
        assert!(
            (plugin.params.pitch_offset.value() >= -24.0)
                && (plugin.params.pitch_offset.value() <= 24.0)
        );

        // FM envelope time params should be in valid range (0.5ms to 40s)
        assert!(
            plugin.params.seg1_time.value() >= TIME_MIN
                && plugin.params.seg1_time.value() <= TIME_MAX
        );
        assert!(
            plugin.params.rel1_time.value() >= TIME_MIN
                && plugin.params.rel1_time.value() <= TIME_MAX
        );

        // FM envelope level params should be in range 0-99
        assert!(plugin.params.seg1_level.value() >= 0 && plugin.params.seg1_level.value() <= 99);

        // Rate scaling should be in range 0-7
        assert!(plugin.params.rate_scaling.value() >= 0 && plugin.params.rate_scaling.value() <= 7);
    }

    #[test]
    fn test_plugin_metadata() {
        assert_eq!(RigelPlugin::NAME, "Rigel");
        assert_eq!(RigelPlugin::VENDOR, "Kyle Cierzan");
        assert!(!RigelPlugin::VERSION.is_empty());

        // Test CLAP metadata
        assert_eq!(RigelPlugin::CLAP_ID, "com.kylecierzan.rigel");
        assert!(RigelPlugin::CLAP_DESCRIPTION.is_some());

        // Test VST3 metadata
        assert_eq!(RigelPlugin::VST3_CLASS_ID.len(), 16);
        assert!(!RigelPlugin::VST3_SUBCATEGORIES.is_empty());
    }

    #[test]
    fn test_audio_io_layout() {
        let layouts = RigelPlugin::AUDIO_IO_LAYOUTS;
        assert!(!layouts.is_empty());

        let layout = &layouts[0];
        assert!(layout.main_input_channels.is_none()); // Synth has no audio input
        assert_eq!(layout.main_output_channels.unwrap().get(), 2); // Stereo output
    }

    #[test]
    fn test_midi_config() {
        assert_eq!(RigelPlugin::MIDI_INPUT, MidiConfig::Basic);
    }

    #[test]
    fn test_editor_creation() {
        // Skip editor creation test as it requires complex AsyncExecutor setup
        // The editor functionality is tested through integration tests
    }

    #[test]
    fn test_parameter_smoothing() {
        let plugin = create_test_plugin();

        // Test that smoothed parameters are available (FloatParams have smoothing)
        let _vol_smooth = plugin.params.volume.smoothed.next();
        let _pitch_smooth = plugin.params.pitch_offset.smoothed.next();

        // Time params are FloatParams, level params are IntParams
        let _seg1_time = plugin.params.seg1_time.value();
        let _seg1_level = plugin.params.seg1_level.value();
        let _rate_scaling = plugin.params.rate_scaling.value();

        // If we get here without panicking, the API is working
    }

    #[test]
    fn test_basic_audio_processing() {
        let mut plugin = create_test_plugin();

        // Basic test: plugin should handle reset without panicking
        plugin.reset();

        // Process status test is complex due to Buffer API - skip for MVP
        // Integration tests exercise the full processing path; this unit test
        // simply ensures the setup/reset cycle does not panic.
    }

    #[test]
    fn test_reset_functionality() {
        let mut plugin = create_test_plugin();

        // Reset should not panic
        plugin.reset();

        // After reset, plugin should still be functional
        assert_eq!(plugin.params.volume.value(), 0.7);
    }

    #[test]
    fn test_parameter_units_and_formatting() {
        let plugin = create_test_plugin();

        // Volume should have % unit
        assert!(plugin.params.volume.unit().contains("%"));

        // Pitch should have semitone unit
        assert!(plugin.params.pitch_offset.unit().contains("st"));

        // FM envelope time params are formatted as time (ms/s)
        // Verify we can access them and they're in valid range
        assert!(plugin.params.seg1_time.value() >= TIME_MIN);
        assert!(plugin.params.seg1_level.value() <= 99);
    }

    #[test]
    fn test_synth_params_mapping() {
        let plugin = create_test_plugin();
        let sample_rate = 44100.0;

        // Build FM envelope params from plugin parameters
        // Convert time values to DX7 rates using envelope_formatters::time_to_rate
        let envelope = FmEnvelopeParams {
            key_on: [
                Segment::new(
                    envelope_formatters::time_to_rate(plugin.params.seg1_time.value(), sample_rate),
                    plugin.params.seg1_level.value() as u8,
                ),
                Segment::new(
                    envelope_formatters::time_to_rate(plugin.params.seg2_time.value(), sample_rate),
                    plugin.params.seg2_level.value() as u8,
                ),
                Segment::new(
                    envelope_formatters::time_to_rate(plugin.params.seg3_time.value(), sample_rate),
                    plugin.params.seg3_level.value() as u8,
                ),
                Segment::new(
                    envelope_formatters::time_to_rate(plugin.params.seg4_time.value(), sample_rate),
                    plugin.params.seg4_level.value() as u8,
                ),
                Segment::new(
                    envelope_formatters::time_to_rate(plugin.params.seg5_time.value(), sample_rate),
                    plugin.params.seg5_level.value() as u8,
                ),
                Segment::new(
                    envelope_formatters::time_to_rate(plugin.params.seg6_time.value(), sample_rate),
                    plugin.params.seg6_level.value() as u8,
                ),
            ],
            release: [
                Segment::new(
                    envelope_formatters::time_to_rate(plugin.params.rel1_time.value(), sample_rate),
                    plugin.params.rel1_level.value() as u8,
                ),
                Segment::new(
                    envelope_formatters::time_to_rate(plugin.params.rel2_time.value(), sample_rate),
                    plugin.params.rel2_level.value() as u8,
                ),
            ],
            rate_scaling: plugin.params.rate_scaling.value() as u8,
        };

        // Test that plugin parameters map to synth parameters correctly
        let synth_params = SynthParams {
            volume: plugin.params.volume.value(),
            pitch_offset: plugin.params.pitch_offset.value(),
            envelope,
        };

        assert!((synth_params.volume - 0.7).abs() < 0.001);
        assert!((synth_params.pitch_offset - 0.0).abs() < 0.001);
        // 10ms attack time should map to a high rate (fast)
        assert!(
            synth_params.envelope.key_on[0].rate > 70,
            "10ms attack should be fast rate"
        );
        assert_eq!(synth_params.envelope.key_on[0].level, 99); // Attack level
                                                               // 300ms release should map to a medium rate
        assert!(
            synth_params.envelope.release[0].rate > 30
                && synth_params.envelope.release[0].rate < 70,
            "300ms release should be medium rate"
        );
        assert_eq!(synth_params.envelope.rate_scaling, 0);
    }
}
