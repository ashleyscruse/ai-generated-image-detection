# Morehouse Research Poster Template

A LaTeX poster template using Morehouse College brand colors (maroon `#840028`, gold `#C1A231`) on a white background. Designed to be print-friendly: only the title bar and section header strips use color, so most of the page prints in black ink.

## Files

- `poster_morehouse.tex` — the template. Pre-filled with NOBLE benchmark example content so you can see how each section is structured. Replace the example content with your own.

## How to compile

You need a working LaTeX install. On macOS, [MacTeX](https://www.tug.org/mactex/) is the standard. On Linux, `texlive-full` from your package manager. On Windows, [MiKTeX](https://miktex.org/).

```bash
cd templates/poster
pdflatex poster_morehouse.tex
pdflatex poster_morehouse.tex   # second pass resolves cross-references
```

Output is `poster_morehouse.pdf` in the same folder. Open it with any PDF viewer to preview.

If you want a different look (better fonts), you can switch to `xelatex` instead of `pdflatex`. The template uses Latin Modern by default which works with both engines.

## How to customize for your project

1. **Title, authors, institute.** Edit the `\title{}`, `\author{}`, and `\institute{}` lines near the top.
2. **Section content.** Each `\block{Title}{Body}` is one panel. Replace the body text. The titles are the dark maroon strips.
3. **Add or remove panels.** Wrap two `\block` calls in `\begin{columns} \column{0.5} ... \column{0.5} ... \end{columns}` to put them side by side. Drop the `columns` wrapper to get a full-width panel.
4. **Add a logo.** Drop a logo file (PNG or PDF) into `../assets/`, then uncomment the `\titlegraphic{}` line near the top.
5. **Change paper size.** A0 landscape is the default. To switch to US conference standard (48"x36" landscape), see the comment block at the top of the `.tex` file.

## Brand rules

- **Maroon** is for the title bar background and section header strips. Don't use it for body text.
- **Gold** is reserved for callout boxes and inner-block accents. Don't use it for body text.
- **Body text stays black.** This keeps the poster legible and print-friendly.
- **Background stays white.** Saves print ink and matches Morehouse brand for printed assets.

## Things to fix before printing

- [ ] Replace placeholder `N` counts with actual numbers from your final dataset
- [ ] Add a real figure or chart in the Results section (the table is a starting point, not a finished result)
- [ ] Update the project repository and dataset card URLs in the Acknowledgments section
- [ ] Confirm author list and affiliation footnotes
- [ ] Drop in your project's specific funding statement
- [ ] Verify the printer's required dimensions (most academic conferences specify size)

## Common issues

**"Undefined control sequence" or font errors when compiling.**
Run `pdflatex` twice. The first pass populates auxiliary files; the second resolves cross-references.

**Font looks rough or non-Morehouse.**
You're using Latin Modern (a free font that ships with TeX Live). Morehouse's official fonts (DIN Pro for headlines, Adobe Caslon Pro for body) are commercial and not bundled. For closer brand match, install the fonts and switch to `xelatex` with `fontspec`. For most poster purposes, Latin Modern looks fine and reads well from across a poster hall.

**"File not found" for `morehouse_logo.png`.**
The logo line is commented out by default. If you uncomment it, drop a logo file into `../assets/morehouse_logo.png` first.

## Sharing with students

This whole folder is portable. Zip `templates/poster/` and send it. Students need a working LaTeX install (see "How to compile" above) and the example content shows them where to put their own writing.
