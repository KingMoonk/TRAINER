import type { NextConfig } from "next";

// A static export: `next build` writes web/out, which the API server serves (docs/rewrite.md, Phase 1).
const nextConfig: NextConfig = {
  output: "export",
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
