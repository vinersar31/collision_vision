import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/collision_vision',
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
