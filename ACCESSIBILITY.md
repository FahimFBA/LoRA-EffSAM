# Accessibility Statement

LoRA-EffSAM is a research code package with a command-line interface, plus a static
project website and documentation. We want these to be usable by as many people as
possible, including people using assistive technology.

## Commitment

* Documentation is written in plain Markdown with a logical heading structure so it works
  with screen readers and text-only readers.
* Figures in the documentation have descriptive captions or alt text, and key results are
  also given as text or tables rather than only as images.
* The website aims to follow the
  [WCAG 2.1](https://www.w3.org/TR/WCAG21/) Level AA guidelines. It was reviewed manually
  (source and computed color contrast, not with a screen reader or automated scanner). It
  has semantic landmarks and headings, a skip link, visible keyboard focus, text
  alternatives for all images, labelled buttons, light and dark themes, and honors the
  reduced-motion setting.
* The CLI uses plain text output and does not rely on color alone to convey information.

## Supported environments

* Command line: any terminal on Linux or WSL2 (training and evaluation need an NVIDIA GPU).
* Documentation and website: current versions of major browsers (Chrome, Firefox, Safari,
  Edge), with or without a screen reader such as NVDA, JAWS or VoiceOver.

## Known limitations

* Some result figures (for example segmentation overlays and attention maps) are
  inherently visual. Where possible the underlying numbers are given in the text or
  tables, but not every figure has a full textual equivalent yet.
* Some figures may rely on color to distinguish classes or curves.
* On narrow screens the top bar shows only the GitHub link; section navigation is by
  scrolling or headings.
* The website loads its fonts from Google Fonts and falls back to system fonts if blocked.
* The website has not been tested with a screen reader or an automated checker.
* Segmentation of medical images is a visual task; the project does not provide a
  graphical interface.

## Reporting accessibility barriers

If you run into a barrier using the documentation, website or CLI, please
[open an issue](https://github.com/FahimFBA/lora-effsam/issues/new/choose) describing the
problem and, if relevant, the assistive technology and browser or terminal you use, or
email fahimbinamin@gmail.com. We aim to respond within 14 days and to fix reported
barriers where practical.
