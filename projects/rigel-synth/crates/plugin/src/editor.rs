use nih_plug::prelude::Editor;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::Arc;

use crate::RigelPluginParams;

/// Default editor state - 600x700 window size for FM envelope controls
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (600, 700))
}

/// Data struct for reactive binding via Lens derive
#[derive(Lens, Clone)]
struct Data {
    params: Arc<RigelPluginParams>,
}

impl Model for Data {}

pub(crate) fn create(
    params: Arc<RigelPluginParams>,
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::default(), move |cx, _| {
        Data {
            params: params.clone(),
        }
        .build(cx);

        VStack::new(cx, |cx| {
            // Header
            Label::new(cx, "Rigel").font_size(24.0);
            Label::new(cx, "FM Wavetable Synthesizer");

            // Master controls
            HStack::new(cx, |cx| {
                VStack::new(cx, |cx| {
                    Label::new(cx, "Volume");
                    ParamSlider::new(cx, Data::params, |params| &params.volume);
                })
                .width(Stretch(1.0));

                VStack::new(cx, |cx| {
                    Label::new(cx, "Pitch");
                    ParamSlider::new(cx, Data::params, |params| &params.pitch_offset);
                })
                .width(Stretch(1.0));
            })
            .col_between(Pixels(10.0));

            // Separator
            Element::new(cx)
                .height(Pixels(1.0))
                .background_color(Color::gray());

            // Key-On Segments Header
            Label::new(cx, "Key-On Segments (Attack/Decay)").font_size(16.0);

            // Segments 1-3 (Attack and first decays)
            HStack::new(cx, |cx| {
                // Segment 1 (Attack)
                VStack::new(cx, |cx| {
                    Label::new(cx, "Seg 1 (Attack)");
                    Label::new(cx, "Rate");
                    ParamSlider::new(cx, Data::params, |params| &params.seg1_rate);
                    Label::new(cx, "Level");
                    ParamSlider::new(cx, Data::params, |params| &params.seg1_level);
                })
                .width(Stretch(1.0));

                // Segment 2 (Decay 1)
                VStack::new(cx, |cx| {
                    Label::new(cx, "Seg 2 (Decay)");
                    Label::new(cx, "Rate");
                    ParamSlider::new(cx, Data::params, |params| &params.seg2_rate);
                    Label::new(cx, "Level");
                    ParamSlider::new(cx, Data::params, |params| &params.seg2_level);
                })
                .width(Stretch(1.0));

                // Segment 3
                VStack::new(cx, |cx| {
                    Label::new(cx, "Seg 3");
                    Label::new(cx, "Rate");
                    ParamSlider::new(cx, Data::params, |params| &params.seg3_rate);
                    Label::new(cx, "Level");
                    ParamSlider::new(cx, Data::params, |params| &params.seg3_level);
                })
                .width(Stretch(1.0));
            })
            .col_between(Pixels(10.0));

            // Segments 4-6 (Later decays and sustain)
            HStack::new(cx, |cx| {
                // Segment 4
                VStack::new(cx, |cx| {
                    Label::new(cx, "Seg 4");
                    Label::new(cx, "Rate");
                    ParamSlider::new(cx, Data::params, |params| &params.seg4_rate);
                    Label::new(cx, "Level");
                    ParamSlider::new(cx, Data::params, |params| &params.seg4_level);
                })
                .width(Stretch(1.0));

                // Segment 5
                VStack::new(cx, |cx| {
                    Label::new(cx, "Seg 5");
                    Label::new(cx, "Rate");
                    ParamSlider::new(cx, Data::params, |params| &params.seg5_rate);
                    Label::new(cx, "Level");
                    ParamSlider::new(cx, Data::params, |params| &params.seg5_level);
                })
                .width(Stretch(1.0));

                // Segment 6 (Sustain)
                VStack::new(cx, |cx| {
                    Label::new(cx, "Seg 6 (Sustain)");
                    Label::new(cx, "Rate");
                    ParamSlider::new(cx, Data::params, |params| &params.seg6_rate);
                    Label::new(cx, "Level");
                    ParamSlider::new(cx, Data::params, |params| &params.seg6_level);
                })
                .width(Stretch(1.0));
            })
            .col_between(Pixels(10.0));

            // Separator
            Element::new(cx)
                .height(Pixels(1.0))
                .background_color(Color::gray());

            // Release Segments Header
            Label::new(cx, "Release Segments").font_size(16.0);

            HStack::new(cx, |cx| {
                // Release 1
                VStack::new(cx, |cx| {
                    Label::new(cx, "Release 1");
                    Label::new(cx, "Rate");
                    ParamSlider::new(cx, Data::params, |params| &params.rel1_rate);
                    Label::new(cx, "Level");
                    ParamSlider::new(cx, Data::params, |params| &params.rel1_level);
                })
                .width(Stretch(1.0));

                // Release 2
                VStack::new(cx, |cx| {
                    Label::new(cx, "Release 2");
                    Label::new(cx, "Rate");
                    ParamSlider::new(cx, Data::params, |params| &params.rel2_rate);
                    Label::new(cx, "Level");
                    ParamSlider::new(cx, Data::params, |params| &params.rel2_level);
                })
                .width(Stretch(1.0));

                // Rate Scaling
                VStack::new(cx, |cx| {
                    Label::new(cx, "Rate Scaling");
                    ParamSlider::new(cx, Data::params, |params| &params.rate_scaling);
                    Label::new(cx, "(Keyboard tracking)");
                })
                .width(Stretch(1.0));
            })
            .col_between(Pixels(10.0));

            ResizeHandle::new(cx);
        })
        .child_space(Pixels(10.0))
        .row_between(Pixels(5.0));
    })
}
