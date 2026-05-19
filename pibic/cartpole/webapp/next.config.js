/** @type {import('next').NextConfig} */
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

module.exports = {
  output: 'export',
  basePath,
  assetPrefix: basePath ? basePath + '/' : '',
  images: { unoptimized: true },
  trailingSlash: true,
};
