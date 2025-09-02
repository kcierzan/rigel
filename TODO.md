# TODO

- [x] Create cargo workspace with crates for crates/dsp, crates/plugin, crates/cli
- [x] Add Cargo.toml with workspace members
- [x] Add rust-toolchain.toml pinned to stable
- [x] Enable LTO/thinlto in workspace profiles
- [x] Create README with build/run instructions
- [x] Set up .gitignore for target/DAW bundles

## Completed ✅

All initial project setup tasks have been completed. The project now has:

- A working `no_std` DSP core with monophonic synthesis
- A CLI tool for generating test audio files
- A headless NIH-plug based plugin with VST3/CLAP support
- Clean project structure with minimal dependencies
- Comprehensive documentation and build instructions

## Next Steps

- [ ] Add wavetable synthesis capabilities to DSP core
- [ ] Implement polyphonic voice management
- [ ] Add audio filters (low-pass, high-pass, band-pass)
- [x] Create NIH-plug based plugin with VST3/CLAP support
- [ ] Develop Iced-based GUI for the plugin
- [ ] Add LFO and modulation system
- [ ] Implement effects processing (reverb, delay, chorus)
- [ ] Add preset management system
