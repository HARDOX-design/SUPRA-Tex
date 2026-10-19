# Example inputs

Use these images with the [image-to-3D example](../README.md#quick-start).
`infer.py` and `scripts/run_example.py` default to `spiny_whelk_shell.png` when
`--input` is omitted.
To select another example, pass `--input assets/<filename>` from the repository root.

RGBA inputs include foreground masks and can use `--use-input-alpha` without RMBG-2.
For the RGB candle jar and cinema signage, omit `--use-input-alpha` and provide
`--rmbg-model /path/to/RMBG-2.0`.

| Example | File | Format |
| --- | --- | --- |
| [Spiny whelk shell (default)](spiny_whelk_shell.png) | `spiny_whelk_shell.png` | RGBA |
| [Pitcher plant](pitcher_plant.png) | `pitcher_plant.png` | RGBA |
| [Banyan bonsai](banyan_bonsai.png) | `banyan_bonsai.png` | RGBA |
| [Potted moth orchid](potted_moth_orchid.png) | `potted_moth_orchid.png` | RGBA |
| [Industrial disconnect switch (`--refine` example)](industrial_disconnect_switch.png) | `industrial_disconnect_switch.png` | RGBA |
| [Metalwork scented candle jar](metalwork_candle_jar.png) | `metalwork_candle_jar.png` | RGB |
| [Ceramic apothecary jar](ceramic_apothecary_jar.png) | `ceramic_apothecary_jar.png` | RGBA |
| [Cinema signage (dual-view example)](cinema_signage.png) | `cinema_signage.png` | RGB |
| [Bronze ding vessel](bronze_ding_vessel.png) | `bronze_ding_vessel.png` | RGBA |
| [Celadon brush washer](celadon_brush_washer.png) | `celadon_brush_washer.png` | RGBA |
| [Amber bird brooch](amber_bird_brooch.png) | `amber_bird_brooch.png` | RGBA |
| [Sheet music archive box](sheet_music_archive_box.png) | `sheet_music_archive_box.png` | RGBA |
| [Ginkgo lidded jar](ginkgo_lidded_jar.png) | `ginkgo_lidded_jar.png` | RGBA |
| [Hammered copper jar](hammered_copper_jar.png) | `hammered_copper_jar.png` | RGBA |

## Texturing example

[`examples/pitcher_plant/`](examples/pitcher_plant) is the default input of
`texturing.py`. It holds the project-page pitcher plant mesh in the canonical
frame (`geometry.npz`, 388,551 vertices and 494,890 faces), its MoGe-2 camera
(`camera.json`), the foreground condition (`condition_rgba.png`), and the
reference-view material maps (`albedo.png`, `mra.png`) predicted by the
released material adapter from `pitcher_plant.png`.
