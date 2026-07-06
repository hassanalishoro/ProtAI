/** PostCSS config — replaces the deprecated @astrojs/tailwind integration.
 *  Astro 6 picks this up automatically; no astro.config.mjs entry needed.
 */
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
