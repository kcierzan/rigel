## Project Overview

wtgen is a Python CLI tool for wavetable generation and processing, specializing in harmonic
wavetable synthesis and bandlimited mipmap generation for alias-free audio playback.

## Development Setup

Use devenv shell for all development tasks:

```sh
devenv shell
```

## Commands

| Command | Description |
|---------|-------------|
| `lint` | Ruff linter |
| `format` | Ruff formatter |
| `typecheck` | ty type checker |
| `test:full` | Parallel pytest with xdist |
| `test:fast` | Single-process pytest with early exit |

Run via: `devenv shell -- <command>`

## Project Structure

- `src/wtgen/cli/` - CLI commands (public API)
- `src/wtgen/dsp/` - DSP processing (waves, mipmaps, filters)
- `src/wtgen/format/` - WTBL file format (reader, writer, validation)

## Completion Requirements

ALWAYS run before considering changes complete:
- `test:full` - all tests must pass
- `typecheck` - ty must pass
- `lint` - ruff must pass

Add tests for any new code.
