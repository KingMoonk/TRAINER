import type { NextConfig } from "next";

// A static export: `next build` writes web/out, which the API server serves (docs/rewrite.md, Phase 1).
const nextConfig: NextConfig = {
  output: "export",
  // /queue/ -> queue/index.html: any static server (the API's, Playwright's) serves a route without knowing about .html.
  trailingSlash: true,
  images: { unoptimized: true },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
