import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  allowedDevOrigins: ["meetingmind.local"],
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
