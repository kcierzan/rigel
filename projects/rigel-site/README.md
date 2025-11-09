# Rigel Site (Placeholder)

This directory will house the public marketing/documentation site for Rigel.
During the repository restructure we stubbed out this project to make the
monorepo layout future-ready.  Until a real implementation lands, this folder
captures plans and expected technology choices so discussions have a single
home.

## Proposed Scope
- Static marketing site that showcases the plugin, wtgen research output, and
  audio demos.
- Documentation hub that links directly into the `rigel-synth` and `wtgen`
  READMEs plus generated API docs in the future.
- Download/build instructions and release notes once binary bundles are shared
  publicly.

## Next Steps
1. Decide on the build stack (Astro, Next.js, Remix, etc.).
2. Stand up a minimal CI workflow that publishes preview builds to the chosen
   hosting provider (Cloudflare Pages, Vercel, Netlify, ...).
3. Wire the site to reference the backend service once it exists so plugin
   users can authenticate/download extra content.
