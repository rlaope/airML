# Brand assets

| File | Use |
|------|-----|
| logo.svg | Main logo with wordmark |
| logo-mark.svg | Just the mark, square viewBox |
| logo-wide.svg | Wide header version |
| favicon.svg | Browser favicon source |
| social-card.svg | Open Graph / Twitter Card source |

## Color tokens

- Foreground: `currentColor` (inherits from page)
- Accent: `#0a84ff` (system blue, matches macOS native)
- Bg dark: `#0a0a0a`
- Bg light: `#fafafa`

## Rasterizing

To produce PNGs (for Open Graph use):

    rsvg-convert -w 1200 -h 630 social-card.svg > social-card.png
    rsvg-convert -w 32 -h 32 favicon.svg > favicon-32.png

## License

All assets are MIT-licensed alongside the rest of airML.
