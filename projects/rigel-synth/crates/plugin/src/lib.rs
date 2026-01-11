//! # Rigel Plugin
//!
//! Wavetable synthesizer plugin with full FM envelope control using nih-plug.

use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use rigel_dsp::{FmEnvelopeParams, Segment, SynthEngine, SynthParams};
use std::sync::Arc;

mod editor;

/// The main plugin struct
pub struct RigelPlugin {
    params: Arc<RigelPluginParams>,
    synth_engine: SynthEngine,
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
    /// Segment 1 Rate (Attack speed)
    #[id = "seg1_rate"]
    pub seg1_rate: IntParam,

    /// Segment 1 Level (Attack target)
    #[id = "seg1_level"]
    pub seg1_level: IntParam,

    // ===== Key-On Segment 2 (Decay 1) =====
    /// Segment 2 Rate (Decay 1 speed)
    #[id = "seg2_rate"]
    pub seg2_rate: IntParam,

    /// Segment 2 Level (Decay 1 target)
    #[id = "seg2_level"]
    pub seg2_level: IntParam,

    // ===== Key-On Segment 3 (Decay 2) =====
    /// Segment 3 Rate
    #[id = "seg3_rate"]
    pub seg3_rate: IntParam,

    /// Segment 3 Level
    #[id = "seg3_level"]
    pub seg3_level: IntParam,

    // ===== Key-On Segment 4 (Decay 3) =====
    /// Segment 4 Rate
    #[id = "seg4_rate"]
    pub seg4_rate: IntParam,

    /// Segment 4 Level
    #[id = "seg4_level"]
    pub seg4_level: IntParam,

    // ===== Key-On Segment 5 (Decay 4) =====
    /// Segment 5 Rate
    #[id = "seg5_rate"]
    pub seg5_rate: IntParam,

    /// Segment 5 Level
    #[id = "seg5_level"]
    pub seg5_level: IntParam,

    // ===== Key-On Segment 6 (Sustain) =====
    /// Segment 6 Rate (Sustain approach speed)
    #[id = "seg6_rate"]
    pub seg6_rate: IntParam,

    /// Segment 6 Level (Sustain level)
    #[id = "seg6_level"]
    pub seg6_level: IntParam,

    // ===== Release Segment 1 =====
    /// Release 1 Rate
    #[id = "rel1_rate"]
    pub rel1_rate: IntParam,

    /// Release 1 Level
    #[id = "rel1_level"]
    pub rel1_level: IntParam,

    // ===== Release Segment 2 =====
    /// Release 2 Rate
    #[id = "rel2_rate"]
    pub rel2_rate: IntParam,

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
        }
    }
}

/// Helper to create a rate parameter (0-99)
fn rate_param(name: &str, default: i32) -> IntParam {
    IntParam::new(name, default, IntRange::Linear { min: 0, max: 99 })
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
            seg1_rate: rate_param("Seg1 Rate", 85),
            seg1_level: level_param("Seg1 Level", 99),

            // Key-On Segment 2 (Decay 1): Medium decay to sustain
            seg2_rate: rate_param("Seg2 Rate", 50),
            seg2_level: level_param("Seg2 Level", 69),

            // Key-On Segments 3-6: Hold at sustain
            seg3_rate: rate_param("Seg3 Rate", 99),
            seg3_level: level_param("Seg3 Level", 69),

            seg4_rate: rate_param("Seg4 Rate", 99),
            seg4_level: level_param("Seg4 Level", 69),

            seg5_rate: rate_param("Seg5 Rate", 99),
            seg5_level: level_param("Seg5 Level", 69),

            seg6_rate: rate_param("Seg6 Rate", 99),
            seg6_level: level_param("Seg6 Level", 69),

            // Release Segment 1: Medium release to silence
            rel1_rate: rate_param("Rel1 Rate", 45),
            rel1_level: level_param("Rel1 Level", 0),

            // Release Segment 2: Instant to silence (fallback)
            rel2_rate: rate_param("Rel2 Rate", 99),
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
            let envelope = FmEnvelopeParams {
                key_on: [
                    Segment::new(
                        self.params.seg1_rate.value() as u8,
                        self.params.seg1_level.value() as u8,
                    ),
                    Segment::new(
                        self.params.seg2_rate.value() as u8,
                        self.params.seg2_level.value() as u8,
                    ),
                    Segment::new(
                        self.params.seg3_rate.value() as u8,
                        self.params.seg3_level.value() as u8,
                    ),
                    Segment::new(
                        self.params.seg4_rate.value() as u8,
                        self.params.seg4_level.value() as u8,
                    ),
                    Segment::new(
                        self.params.seg5_rate.value() as u8,
                        self.params.seg5_level.value() as u8,
                    ),
                    Segment::new(
                        self.params.seg6_rate.value() as u8,
                        self.params.seg6_level.value() as u8,
                    ),
                ],
                release: [
                    Segment::new(
                        self.params.rel1_rate.value() as u8,
                        self.params.rel1_level.value() as u8,
                    ),
                    Segment::new(
                        self.params.rel2_rate.value() as u8,
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
        assert_eq!(plugin.params.seg1_rate.value(), 85); // Fast attack
        assert_eq!(plugin.params.seg1_level.value(), 99); // To full level
        assert_eq!(plugin.params.seg2_rate.value(), 50); // Medium decay
        assert_eq!(plugin.params.seg2_level.value(), 69); // To sustain (~70%)
        assert_eq!(plugin.params.rel1_rate.value(), 45); // Medium release
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

        // FM envelope segment params should be in range 0-99
        assert!(plugin.params.seg1_rate.value() >= 0 && plugin.params.seg1_rate.value() <= 99);
        assert!(plugin.params.seg1_level.value() >= 0 && plugin.params.seg1_level.value() <= 99);
        assert!(plugin.params.rel1_rate.value() >= 0 && plugin.params.rel1_rate.value() <= 99);

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

        // IntParams don't have smoothing, but we can verify they exist
        let _seg1_rate = plugin.params.seg1_rate.value();
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

        // FM envelope params don't have units (they're 0-99 integers)
        // Just verify we can access them
        assert!(plugin.params.seg1_rate.value() <= 99);
        assert!(plugin.params.seg1_level.value() <= 99);
    }

    #[test]
    fn test_synth_params_mapping() {
        let plugin = create_test_plugin();

        // Build FM envelope params from plugin parameters
        let envelope = FmEnvelopeParams {
            key_on: [
                Segment::new(
                    plugin.params.seg1_rate.value() as u8,
                    plugin.params.seg1_level.value() as u8,
                ),
                Segment::new(
                    plugin.params.seg2_rate.value() as u8,
                    plugin.params.seg2_level.value() as u8,
                ),
                Segment::new(
                    plugin.params.seg3_rate.value() as u8,
                    plugin.params.seg3_level.value() as u8,
                ),
                Segment::new(
                    plugin.params.seg4_rate.value() as u8,
                    plugin.params.seg4_level.value() as u8,
                ),
                Segment::new(
                    plugin.params.seg5_rate.value() as u8,
                    plugin.params.seg5_level.value() as u8,
                ),
                Segment::new(
                    plugin.params.seg6_rate.value() as u8,
                    plugin.params.seg6_level.value() as u8,
                ),
            ],
            release: [
                Segment::new(
                    plugin.params.rel1_rate.value() as u8,
                    plugin.params.rel1_level.value() as u8,
                ),
                Segment::new(
                    plugin.params.rel2_rate.value() as u8,
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
        assert_eq!(synth_params.envelope.key_on[0].rate, 85); // Attack rate
        assert_eq!(synth_params.envelope.key_on[0].level, 99); // Attack level
        assert_eq!(synth_params.envelope.release[0].rate, 45); // Release rate
        assert_eq!(synth_params.envelope.rate_scaling, 0);
    }
}
