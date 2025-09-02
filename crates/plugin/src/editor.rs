use nih_plug::prelude::{Editor, GuiContext};
use nih_plug_iced::widgets as nih_widgets;
use nih_plug_iced::*;
use std::sync::Arc;

use crate::RigelPluginParams;

// Default editor state - compact size for minimal UI
pub(crate) fn default_state() -> Arc<IcedState> {
    IcedState::from_size(300, 400)
}

pub(crate) fn create(
    params: Arc<RigelPluginParams>,
    editor_state: Arc<IcedState>,
) -> Option<Box<dyn Editor>> {
    create_iced_editor::<RigelEditor>(editor_state, params)
}

struct RigelEditor {
    params: Arc<RigelPluginParams>,
    context: Arc<dyn GuiContext>,

    // Widget states
    volume_slider_state: nih_widgets::param_slider::State,
    pitch_slider_state: nih_widgets::param_slider::State,
    attack_slider_state: nih_widgets::param_slider::State,
    decay_slider_state: nih_widgets::param_slider::State,
    sustain_slider_state: nih_widgets::param_slider::State,
    release_slider_state: nih_widgets::param_slider::State,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    /// Update a parameter's value.
    ParamUpdate(nih_widgets::ParamMessage),
}

impl IcedEditor for RigelEditor {
    type Executor = executor::Default;
    type Message = Message;
    type InitializationFlags = Arc<RigelPluginParams>;

    fn new(
        params: Self::InitializationFlags,
        context: Arc<dyn GuiContext>,
    ) -> (Self, Command<Self::Message>) {
        let editor = RigelEditor {
            params,
            context,

            volume_slider_state: Default::default(),
            pitch_slider_state: Default::default(),
            attack_slider_state: Default::default(),
            decay_slider_state: Default::default(),
            sustain_slider_state: Default::default(),
            release_slider_state: Default::default(),
        };

        (editor, Command::none())
    }

    fn context(&self) -> &dyn GuiContext {
        self.context.as_ref()
    }

    fn update(
        &mut self,
        _window: &mut WindowQueue,
        message: Self::Message,
    ) -> Command<Self::Message> {
        match message {
            Message::ParamUpdate(message) => self.handle_param_message(message),
        }

        Command::none()
    }

    fn view(&mut self) -> Element<'_, Self::Message> {
        Column::new()
            .align_items(Alignment::Center)
            .spacing(10)
            .padding(20)
            .push(
                Text::new("Rigel")
                    .font(assets::NOTO_SANS_LIGHT)
                    .size(32)
                    .height(40.into())
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                Text::new("Wavetable Synthesizer")
                    .size(14)
                    .height(20.into())
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(Space::with_height(10.into()))
            // Volume control
            .push(
                Column::new()
                    .spacing(5)
                    .push(Text::new("Volume").size(12))
                    .push(
                        nih_widgets::ParamSlider::new(
                            &mut self.volume_slider_state,
                            &self.params.volume,
                        )
                        .map(Message::ParamUpdate),
                    ),
            )
            // Pitch control
            .push(
                Column::new()
                    .spacing(5)
                    .push(Text::new("Pitch").size(12))
                    .push(
                        nih_widgets::ParamSlider::new(
                            &mut self.pitch_slider_state,
                            &self.params.pitch_offset,
                        )
                        .map(Message::ParamUpdate),
                    ),
            )
            // Envelope section
            .push(
                Text::new("Envelope")
                    .size(16)
                    .height(30.into())
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            // Attack
            .push(
                Column::new()
                    .spacing(5)
                    .push(Text::new("Attack").size(12))
                    .push(
                        nih_widgets::ParamSlider::new(
                            &mut self.attack_slider_state,
                            &self.params.env_attack,
                        )
                        .map(Message::ParamUpdate),
                    ),
            )
            // Decay
            .push(
                Column::new()
                    .spacing(5)
                    .push(Text::new("Decay").size(12))
                    .push(
                        nih_widgets::ParamSlider::new(
                            &mut self.decay_slider_state,
                            &self.params.env_decay,
                        )
                        .map(Message::ParamUpdate),
                    ),
            )
            // Sustain
            .push(
                Column::new()
                    .spacing(5)
                    .push(Text::new("Sustain").size(12))
                    .push(
                        nih_widgets::ParamSlider::new(
                            &mut self.sustain_slider_state,
                            &self.params.env_sustain,
                        )
                        .map(Message::ParamUpdate),
                    ),
            )
            // Release
            .push(
                Column::new()
                    .spacing(5)
                    .push(Text::new("Release").size(12))
                    .push(
                        nih_widgets::ParamSlider::new(
                            &mut self.release_slider_state,
                            &self.params.env_release,
                        )
                        .map(Message::ParamUpdate),
                    ),
            )
            .into()
    }

    fn background_color(&self) -> nih_plug_iced::Color {
        nih_plug_iced::Color {
            r: 0.15,
            g: 0.15,
            b: 0.20,
            a: 1.0,
        }
    }
}
