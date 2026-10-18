# Example inputs

Use these images with the [image-to-3D example](../README.md#image-to-3d).
`infer.py` defaults to `pitcher_plant.png` when `--input` is omitted.
To select another example, pass `--input assets/<filename>` from the repository root.

RGBA inputs include foreground masks and can use `--use-input-alpha` without RMBG-2.
For the RGB candle jar, omit `--use-input-alpha` and provide
`--rmbg-model /path/to/RMBG-2.0`.

| Example | File | Format |
| --- | --- | --- |
| [Pitcher plant (default)](pitcher_plant.png) | `pitcher_plant.png` | RGBA |
| [Spiny whelk shell](spiny_whelk_shell.png) | `spiny_whelk_shell.png` | RGBA |
| [Banyan bonsai](banyan_bonsai.png) | `banyan_bonsai.png` | RGBA |
| [Potted moth orchid](potted_moth_orchid.png) | `potted_moth_orchid.png` | RGBA |
| [Motocross helmet](motocross_helmet.png) | `motocross_helmet.png` | RGBA |
| [Compact excavator](compact_excavator.png) | `compact_excavator.png` | RGBA |
| [Industrial disconnect switch](industrial_disconnect_switch.png) | `industrial_disconnect_switch.png` | RGBA |
| [Metalwork scented candle jar](metalwork_candle_jar.png) | `metalwork_candle_jar.png` | RGB |
| [Ceramic apothecary jar](ceramic_apothecary_jar.png) | `ceramic_apothecary_jar.png` | RGBA |
