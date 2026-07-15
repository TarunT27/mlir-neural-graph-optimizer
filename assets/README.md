# Visual assets

- `hero.png` is an AI-generated project-bound raster created with OpenAI's built-in image
  generation tool on 2026-07-15.
- `before.svg` and `after.svg` are deterministic outputs of `ng-opt visualize`; regenerate them
  from `examples/tiny_mlp.mlir` using the commands below.

```bash
ng-opt visualize examples/tiny_mlp.mlir -o assets/before.svg
ng-opt visualize examples/tiny_mlp.mlir --optimize -o assets/after.svg
```

## Hero prompt

```text
Use case: stylized-concept
Asset type: wide GitHub repository README hero banner
Primary request: an elegant abstract visualization of a neural-network computation graph
flowing through a compiler optimization pipeline, where many tensor nodes and connections on
the left converge into a smaller, cleaner fused graph on the right
Scene/backdrop: deep midnight navy technical space with subtle grid depth and restrained
luminous particles
Subject: geometric tensor blocks, graph nodes, precise connecting paths, and a central compiler
transformation prism; visually communicate graph simplification and operator fusion
Style/medium: premium cinematic 3D technical illustration, polished but credible, crisp
materials and clean silhouettes
Composition/framing: very wide landscape banner, balanced center-weighted flow, generous safe
margins, readable at GitHub README width
Lighting/mood: cool cyan and electric violet edge lighting with a small warm amber accent at the
fused output, confident and sophisticated
Color palette: navy, slate, cyan, violet, restrained amber
Constraints: no text, no letters, no logos, no trademarks, no code snippets, no charts, no
watermark; avoid clutter; do not imply benchmark numbers or speedup claims
```
