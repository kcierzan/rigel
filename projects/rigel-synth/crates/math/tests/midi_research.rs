//! Research: Frequency accuracy analysis for exp2 MIDI calculations
//!
//! This test analyzes the practical frequency accuracy of fast_exp2
//! for MIDI-to-frequency conversion across the full MIDI range.

use rigel_math::simd::fast_exp2;
use rigel_math::{DefaultSimdVector, SimdVector};

#[test]
fn analyze_midi_frequency_accuracy() {
    println!("\n=== MIDI Frequency Accuracy Analysis ===\n");

    // MIDI to frequency formula: freq = 440 * 2^((midi - 69) / 12)
    let a4_midi = 69.0;
    let a4_freq = 440.0;

    let mut max_freq_error_hz = 0.0f32;
    let mut max_freq_error_cents = 0.0f32;
    let mut max_relative_error = 0.0f32;
    let mut worst_midi_note = 0;

    println!("MIDI Note | Expected Hz | Actual Hz   | Error (Hz) | Error (cents) | Rel Error (%)");
    println!("----------|-------------|-------------|------------|---------------|---------------");

    // Test every MIDI note from 0 to 127
    for midi in 0..=127 {
        // Reference calculation using libm
        let semitones_from_a4 = (midi - a4_midi as i32) as f32;
        let octaves = semitones_from_a4 / 12.0;
        let expected_freq = a4_freq * libm::exp2f(octaves);

        // Our fast_exp2 calculation
        let octaves_vec = DefaultSimdVector::splat(octaves);
        let ratio_vec = fast_exp2(octaves_vec);
        let freq_vec = ratio_vec.mul(DefaultSimdVector::splat(a4_freq));
        let actual_freq = freq_vec.horizontal_sum() / DefaultSimdVector::LANES as f32;

        // Calculate errors
        let freq_error_hz = (actual_freq - expected_freq).abs();
        let relative_error = freq_error_hz / expected_freq;

        // Calculate error in cents: cents = 1200 * log2(actual / expected)
        let freq_ratio = actual_freq / expected_freq;
        let error_cents = 1200.0 * libm::log2f(freq_ratio);
        let error_cents_abs = error_cents.abs();

        // Track maximums
        if freq_error_hz > max_freq_error_hz {
            max_freq_error_hz = freq_error_hz;
        }
        if error_cents_abs > max_freq_error_cents {
            max_freq_error_cents = error_cents_abs;
            worst_midi_note = midi;
        }
        if relative_error > max_relative_error {
            max_relative_error = relative_error;
        }

        // Print every 12th note (every octave) + extremes
        if midi % 12 == 0 || midi == 21 || midi == 108 || midi == 127 {
            println!(
                "{:>9} | {:>11.3} | {:>11.3} | {:>10.4} | {:>13.4} | {:>13.6}",
                midi,
                expected_freq,
                actual_freq,
                freq_error_hz,
                error_cents,
                relative_error * 100.0
            );
        }
    }

    println!("\n=== Summary ===");
    println!("Maximum frequency error: {:.6} Hz", max_freq_error_hz);
    println!(
        "Maximum error in cents: {:.6} cents (at MIDI note {})",
        max_freq_error_cents, worst_midi_note
    );
    println!("Maximum relative error: {:.8}%", max_relative_error * 100.0);
    println!("\n=== Musical Context ===");
    println!("Just-noticeable difference (JND): ~5-6 cents");
    println!("\"In tune\" threshold: ~1 cent");
    println!("Our maximum error: {:.6} cents", max_freq_error_cents);
    println!(
        "\nConclusion: Error is {:.1}x BELOW the \"in tune\" threshold",
        1.0 / max_freq_error_cents
    );
    println!(
        "           Error is {:.1}x BELOW the just-noticeable difference",
        5.0 / max_freq_error_cents
    );

    // Verify error is acceptable
    assert!(
        max_freq_error_cents < 1.0,
        "Frequency error should be < 1 cent for musical accuracy"
    );
}

#[test]
fn analyze_exp2_error_pattern() {
    println!("\n=== exp2 Error Pattern Analysis ===\n");
    println!("Analyzing how error varies within fractional range [0, 1)");
    println!("This shows the error pattern that repeats every octave.\n");

    println!("x (octaves) | fast_exp2 | libm::exp2f | Rel Error (%) | Error repeats every 1.0");
    println!("------------|-----------|-------------|---------------|---------------------------");

    let mut errors = Vec::new();

    // Sample the fractional part finely
    for i in 0..100 {
        let x = i as f32 * 0.01; // 0.00, 0.01, 0.02, ..., 0.99
        let x_vec = DefaultSimdVector::splat(x);
        let result = fast_exp2(x_vec);
        let fast_result = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let reference = libm::exp2f(x);
        let relative_error = ((fast_result - reference) / reference).abs();

        errors.push((x, relative_error));

        // Print every 10th sample
        if i % 10 == 0 {
            println!(
                "{:>11.2} | {:>9.6} | {:>11.6} | {:>13.6} |",
                x,
                fast_result,
                reference,
                relative_error * 100.0
            );
        }
    }

    // Find min and max errors
    let min_error = errors.iter().map(|(_, e)| *e).fold(f32::INFINITY, f32::min);
    let max_error = errors
        .iter()
        .map(|(_, e)| *e)
        .fold(f32::NEG_INFINITY, f32::max);
    let avg_error = errors.iter().map(|(_, e)| *e).sum::<f32>() / errors.len() as f32;

    let max_error_x = errors
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let min_error_x = errors
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("\n=== Error Statistics ===");
    println!(
        "Minimum error: {:.8}% at x = {:.2}",
        min_error * 100.0,
        min_error_x.0
    );
    println!(
        "Maximum error: {:.8}% at x = {:.2}",
        max_error * 100.0,
        max_error_x.0
    );
    println!("Average error: {:.8}%", avg_error * 100.0);
    println!("Error range: {:.8}%", (max_error - min_error) * 100.0);
    println!(
        "\nConclusion: Error varies {:.2}x across the fractional range",
        max_error / min_error
    );
    println!("            This pattern repeats identically every octave.");
}

#[test]
fn analyze_error_stability_across_octaves() {
    println!("\n=== Error Stability Across Octaves ===\n");
    println!("Testing if error pattern is identical at different octaves");
    println!("(Testing x=0.5, x=1.5, x=2.5, etc. - same fractional part)\n");

    let fractional_part = 0.5; // Test at the same fractional position
    println!("Octave | x value | fast_exp2 | Rel Error (%) | Error should be identical");
    println!("-------|---------|-----------|---------------|---------------------------");

    let mut errors = Vec::new();

    for octave in -5..=5 {
        let x = octave as f32 + fractional_part;
        let x_vec = DefaultSimdVector::splat(x);
        let result = fast_exp2(x_vec);
        let fast_result = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let reference = libm::exp2f(x);
        let relative_error = ((fast_result - reference) / reference).abs();

        errors.push(relative_error);

        println!(
            "{:>6} | {:>7.1} | {:>9.6} | {:>13.8} |",
            octave,
            x,
            fast_result,
            relative_error * 100.0
        );
    }

    // Check error consistency
    let error_variance = {
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        let variance = errors.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / errors.len() as f32;
        variance
    };

    println!("\n=== Stability Analysis ===");
    println!("Error variance: {:.2e}", error_variance);
    println!(
        "Conclusion: Error is {} stable across octaves",
        if error_variance < 1e-10 {
            "PERFECTLY"
        } else {
            "reasonably"
        }
    );
    println!("            (Fractional part determines error, not absolute value)");
}
