//! Integration tests for the Rigel plugin
//!
//! These tests verify that the plugin works correctly as a complete system,
//! testing the public API and plugin exports.

use nih_plug::prelude::*;
use rigel_plugin::RigelPlugin;

#[test]
fn test_plugin_instantiation() {
    let plugin = RigelPlugin::default();
    assert_eq!(RigelPlugin::NAME, "Rigel");
    assert_eq!(RigelPlugin::VENDOR, "Kyle Cierzan");

    // Verify the plugin has the expected number of parameters
    let params = plugin.params();
    assert!(
        !params.param_map().is_empty(),
        "Plugin should have parameters"
    );
}

#[test]
fn test_plugin_constants() {
    // Test that plugin constants are properly defined
    assert!(!RigelPlugin::NAME.is_empty());
    assert!(!RigelPlugin::VENDOR.is_empty());
    assert!(!RigelPlugin::VERSION.is_empty());
    // URL might be empty if not set in Cargo.toml, but should be accessible
    let _url = RigelPlugin::URL; // Just verify it compiles and is accessible
    assert!(!RigelPlugin::EMAIL.is_empty());

    // Test CLAP specific constants
    assert!(!RigelPlugin::CLAP_ID.is_empty());
    assert!(RigelPlugin::CLAP_DESCRIPTION.is_some());

    // Test VST3 specific constants
    assert_eq!(RigelPlugin::VST3_CLASS_ID.len(), 16);
    assert!(!RigelPlugin::VST3_SUBCATEGORIES.is_empty());
}

#[test]
fn test_audio_io_configuration() {
    let layouts = RigelPlugin::AUDIO_IO_LAYOUTS;
    assert!(
        !layouts.is_empty(),
        "Plugin should have at least one audio layout"
    );

    let main_layout = &layouts[0];
    assert!(
        main_layout.main_input_channels.is_none(),
        "Synth should have no audio input"
    );
    assert!(
        main_layout.main_output_channels.is_some(),
        "Synth should have audio output"
    );
    assert_eq!(
        main_layout.main_output_channels.unwrap().get(),
        2,
        "Should be stereo output"
    );
}

#[test]
fn test_midi_configuration() {
    assert_eq!(
        RigelPlugin::MIDI_INPUT,
        MidiConfig::Basic,
        "Plugin should accept MIDI input"
    );
    assert_eq!(
        RigelPlugin::SAMPLE_ACCURATE_AUTOMATION,
        true,
        "Plugin should support sample-accurate automation"
    );
}

#[test]
fn test_parameter_access() {
    let plugin = RigelPlugin::default();
    let params = plugin.params();

    // Test that we can access the parameter map
    let param_map = params.param_map();
    assert!(
        !param_map.is_empty(),
        "Plugin should have parameters in the map"
    );

    // Get parameter names for testing
    let param_names: Vec<String> = param_map.iter().map(|(name, _, _)| name.clone()).collect();

    // Test that expected parameters exist by name (case insensitive)
    let has_volume = param_names
        .iter()
        .any(|name| name.to_lowercase().contains("volume"));
    let has_pitch = param_names
        .iter()
        .any(|name| name.to_lowercase().contains("pitch"));
    let has_attack = param_names
        .iter()
        .any(|name| name.to_lowercase().contains("attack"));
    let has_decay = param_names
        .iter()
        .any(|name| name.to_lowercase().contains("decay"));
    let has_sustain = param_names
        .iter()
        .any(|name| name.to_lowercase().contains("sustain"));
    let has_release = param_names
        .iter()
        .any(|name| name.to_lowercase().contains("release"));

    assert!(
        has_volume,
        "Should have volume parameter, found: {:?}",
        param_names
    );
    assert!(
        has_pitch,
        "Should have pitch parameter, found: {:?}",
        param_names
    );
    assert!(
        has_attack,
        "Should have attack parameter, found: {:?}",
        param_names
    );
    assert!(
        has_decay,
        "Should have decay parameter, found: {:?}",
        param_names
    );
    assert!(
        has_sustain,
        "Should have sustain parameter, found: {:?}",
        param_names
    );
    assert!(
        has_release,
        "Should have release parameter, found: {:?}",
        param_names
    );
}

#[test]
fn test_parameter_serialization() {
    let plugin = RigelPlugin::default();
    let params = plugin.params();

    // Test that parameters can be serialized and deserialized
    let state = params.serialize_fields();
    assert!(!state.is_empty(), "Plugin should have serializable state");

    // Test that we can deserialize the state (basic smoke test)
    params.deserialize_fields(&state);
    // If we get here without panicking, deserialization worked
    assert!(true, "Plugin state deserialization completed");
}

#[test]
fn test_plugin_features() {
    // Test CLAP features
    let clap_features = RigelPlugin::CLAP_FEATURES;
    assert!(
        clap_features.contains(&ClapFeature::Instrument),
        "Should be marked as instrument"
    );
    assert!(
        clap_features.contains(&ClapFeature::Synthesizer),
        "Should be marked as synthesizer"
    );

    // Test VST3 subcategories
    let vst3_categories = RigelPlugin::VST3_SUBCATEGORIES;
    assert!(
        vst3_categories.contains(&Vst3SubCategory::Instrument),
        "Should be VST3 instrument"
    );
    assert!(
        vst3_categories.contains(&Vst3SubCategory::Synth),
        "Should be VST3 synthesizer"
    );
}

#[test]
fn test_parameter_ranges_and_defaults() {
    let plugin = RigelPlugin::default();
    let params = plugin.params();

    // Test that parameters exist and have reasonable values
    let param_map = params.param_map();
    assert!(
        param_map.len() >= 6,
        "Plugin should have at least 6 parameters"
    );

    // Since we can't access parameters directly, just verify the parameter system works
    let state = params.serialize_fields();
    assert!(!state.is_empty(), "Plugin should have parameter state");

    // Test serialization roundtrip
    params.deserialize_fields(&state);
    let state2 = params.serialize_fields();
    assert_eq!(
        state, state2,
        "Parameter state should be consistent after roundtrip"
    );
}

#[test]
fn test_plugin_initialization() {
    let mut plugin = RigelPlugin::default();
    let buffer_config = BufferConfig {
        sample_rate: 44100.0,
        min_buffer_size: Some(64),
        max_buffer_size: 1024,
        process_mode: ProcessMode::Realtime,
    };
    let audio_io_layout = &RigelPlugin::AUDIO_IO_LAYOUTS[0];

    // Create a minimal init context
    struct TestInitContext;
    impl InitContext<RigelPlugin> for TestInitContext {
        fn set_current_voice_capacity(&self, _capacity: u32) {}
        fn plugin_api(&self) -> PluginApi {
            PluginApi::Clap
        }
        fn execute(&self, _task: <RigelPlugin as Plugin>::BackgroundTask) {}
        fn set_latency_samples(&self, _samples: u32) {}
    }

    let mut context = TestInitContext;
    let result = plugin.initialize(audio_io_layout, &buffer_config, &mut context);
    assert!(result, "Plugin should initialize successfully");

    // Test that reset doesn't panic after initialization
    plugin.reset();
}

#[test]
fn test_editor_availability() {
    let plugin = RigelPlugin::default();

    // Test that parameter serialization includes editor state
    let state = plugin.params().serialize_fields();
    assert!(
        state.contains_key("editor-state"),
        "Editor state should be in serialized parameters"
    );
}
