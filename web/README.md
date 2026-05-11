# podcast-transcriber website

Documentation and landing website for podcast-transcriber. Built with Astro + Tailwind CSS.

## Requirements

- Node.js 18.17.1+
- npm

## Development

```bash
cd web
npm install
npm run dev
```

Opens at http://localhost:4321

## Build

```bash
cd web
npm run build
```

Output goes to `web/dist/`. Serve with `npm run preview` for local preview.

## Structure

- `src/layouts/Layout.astro` — shared layout, nav, footer
- `src/pages/index.astro` — home page
- `src/pages/features.astro` — features overview
- `src/pages/getting-started.astro` — setup, CLI reference
- `src/pages/architecture.astro` — architecture, roadmap

## Notes

- Website is isolated inside `web/` — no Python dependencies
- Content sourced from `../FEATURES.md`
- Dark mode by default (Tailwind `class` strategy)
- Static output only — no server-side rendering
