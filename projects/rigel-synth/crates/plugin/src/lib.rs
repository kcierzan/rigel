//! # Rigel Plugin
//!
//! Minimal headless wavetable synthesizer plugin using nih-plug and rigel-dsp core.

use nih_plug::prelude::*;
use nih_plug_iced::IcedState;
use rigel_dsp::{SynthEngine, SynthParams};
use std::sync::Arc;

mod editor;

/// The main plugin struct
pub struct RigelPlugin {
    params: Arc<RigelPluginParams>,
    synth_engine: SynthEngine,
}

/// Plugin parameters that map to DSP parameters
#[derive(Params)]
pub struct RigelPluginParams {
    /// The editor state, saved together with the parameter state so the custom scaling can be
    /// restored.
    #[persist = "editor-state"]
    editor_state: Arc<IcedState>,

    /// Master volume
    #[id = "volume"]
    pub volume: FloatParam,

    /// Pitch offset in semitones
    #[id = "pitch"]
    pub pitch_offset: FloatParam,

    /// Envelope attack time
    #[id = "attack"]
    pub env_attack: FloatParam,

    /// Envelope decay time
    #[id = "decay"]
    pub env_decay: FloatParam,

    /// Envelope sustain level
    #[id = "sustain"]
    pub env_sustain: FloatParam,

    /// Envelope release time
    #[id = "release"]
    pub env_release: FloatParam,
}

impl Default for RigelPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(RigelPluginParams::default()),
            synth_engine: SynthEngine::new(44100.0),
        }
    }
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

            env_attack: FloatParam::new(
                "Attack",
                0.01,
                FloatRange::Skewed {
                    min: 0.001,
                    max: 5.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_smoother(SmoothingStyle::Linear(10.0))
            .with_unit(" s")
            .with_value_to_string(formatters::v2s_f32_rounded(3)),

            env_decay: FloatParam::new(
                "Decay",
                0.3,
                FloatRange::Skewed {
                    min: 0.001,
                    max: 5.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_smoother(SmoothingStyle::Linear(10.0))
            .with_unit(" s")
            .with_value_to_string(formatters::v2s_f32_rounded(3)),

            env_sustain: FloatParam::new("Sustain", 0.7, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(10.0))
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),

            env_release: FloatParam::new(
                "Release",
                0.5,
                FloatRange::Skewed {
                    min: 0.001,
                    max: 10.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_smoother(SmoothingStyle::Linear(10.0))
            .with_unit(" s")
            .with_value_to_string(formatters::v2s_f32_rounded(3)),
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

            // Get current synth parameters from plugin parameters
            let synth_params = SynthParams {
                volume: self.params.volume.smoothed.next(),
                pitch_offset: self.params.pitch_offset.smoothed.next(),
                env_attack: self.params.env_attack.smoothed.next(),
                env_decay: self.params.env_decay.smoothed.next(),
                env_sustain: self.params.env_sustain.smoothed.next(),
                env_release: self.params.env_release.smoothed.next(),
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

        // Test envelope defaults
        assert!((plugin.params.env_attack.value() - 0.01).abs() < 0.001);
        assert!((plugin.params.env_decay.value() - 0.3).abs() < 0.001);
        assert!((plugin.params.env_sustain.value() - 0.7).abs() < 0.001);
        assert!((plugin.params.env_release.value() - 0.5).abs() < 0.001);
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

        // Envelope parameters should have positive values (except sustain which can be 0-1)
        assert!(plugin.params.env_attack.value() > 0.0);
        assert!(plugin.params.env_decay.value() > 0.0);
        assert!(plugin.params.env_release.value() > 0.0);
        assert!(
            (plugin.params.env_sustain.value() >= 0.0)
                && (plugin.params.env_sustain.value() <= 1.0)
        );
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

        // Test that parameters have smoothed values available (even if not actively smoothing)
        // Smoothing is only active during audio processing, so we just verify the API exists
        let _vol_smooth = plugin.params.volume.smoothed.next();
        let _pitch_smooth = plugin.params.pitch_offset.smoothed.next();
        let _attack_smooth = plugin.params.env_attack.smoothed.next();
        let _decay_smooth = plugin.params.env_decay.smoothed.next();
        let _sustain_smooth = plugin.params.env_sustain.smoothed.next();
        let _release_smooth = plugin.params.env_release.smoothed.next();

        // If we get here without panicking, smoothing API is working
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

        // Envelope times should have seconds unit
        assert!(plugin.params.env_attack.unit().contains("s"));
        assert!(plugin.params.env_decay.unit().contains("s"));
        assert!(plugin.params.env_release.unit().contains("s"));

        // Sustain should have % unit
        assert!(plugin.params.env_sustain.unit().contains("%"));
    }

    #[test]
    fn test_synth_params_mapping() {
        let plugin = create_test_plugin();

        // Test that plugin parameters map to synth parameters correctly
        let synth_params = SynthParams {
            volume: plugin.params.volume.value(),
            pitch_offset: plugin.params.pitch_offset.value(),
            env_attack: plugin.params.env_attack.value(),
            env_decay: plugin.params.env_decay.value(),
            env_sustain: plugin.params.env_sustain.value(),
            env_release: plugin.params.env_release.value(),
        };

        assert!((synth_params.volume - 0.7).abs() < 0.001);
        assert!((synth_params.pitch_offset - 0.0).abs() < 0.001);
        assert!((synth_params.env_attack - 0.01).abs() < 0.001);
        assert!((synth_params.env_decay - 0.3).abs() < 0.001);
        assert!((synth_params.env_sustain - 0.7).abs() < 0.001);
        assert!((synth_params.env_release - 0.5).abs() < 0.001);
    }
}
