# podcast-transcriber website

Documentation and landing website for podcast-transcriber. Built with Astro 7 and Tailwind CSS 4 (via `@tailwindcss/vite`).

## Requirements

- Node.js 22.12.0+ (the Astro 7 floor, enforced via `engines` in `package.json`)
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
