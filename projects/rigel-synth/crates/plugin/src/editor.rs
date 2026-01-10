use nih_plug::prelude::Editor;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::Arc;

use crate::RigelPluginParams;

/// Default editor state - 300x400 window size
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (300, 400))
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
            Label::new(cx, "Rigel");
            Label::new(cx, "Wavetable Synthesizer");

            Label::new(cx, "Volume");
            ParamSlider::new(cx, Data::params, |params| &params.volume);

            Label::new(cx, "Pitch");
            ParamSlider::new(cx, Data::params, |params| &params.pitch_offset);

            Label::new(cx, "Envelope");

            Label::new(cx, "Attack");
            ParamSlider::new(cx, Data::params, |params| &params.env_attack);

            Label::new(cx, "Decay");
            ParamSlider::new(cx, Data::params, |params| &params.env_decay);

            Label::new(cx, "Sustain");
            ParamSlider::new(cx, Data::params, |params| &params.env_sustain);

            Label::new(cx, "Release");
            ParamSlider::new(cx, Data::params, |params| &params.env_release);

            ResizeHandle::new(cx);
        })
        .child_space(Pixels(10.0))
        .row_between(Pixels(5.0));
    })
}
