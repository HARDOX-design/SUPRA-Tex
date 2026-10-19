# Third-Party Notices

SUPRA-Tex's original source code is licensed under the [MIT License](LICENSE).
This repository also includes third-party code under MIT, Apache-2.0, and the
DINOv3 License. Those components remain under their own terms, which are mapped
below; the full texts are reproduced under [License texts](#license-texts).

## Microsoft and Tencent MIT components

The sparse tensor modules, flow/autoencoder infrastructure, mesh representations,
rendering utilities, and inherited pipeline/training infrastructure include code
from Microsoft TRELLIS/TRELLIS.2 and Tencent's MIT-licensed 3D implementation.
Applicable upstream copyrights and full permission notices are retained in:

- [Microsoft MIT license](#microsoft-mit): Copyright (c) Microsoft Corporation.
- [Tencent MIT license](#tencent-mit): Copyright (c) 2026 Tencent.

The SUPRA-Tex package and method names describe this project; they do not change
ownership of incorporated upstream code.

`supratex/inference/mesh_export.py` adapts the O-Voxel exporter from Microsoft
TRELLIS.2 under the Microsoft MIT license above. Supra-Tex replaces its texture
conversion and gap-filling stage with packed GPU quantization and grouped Telea
operations; mesh processing and texture baking retain the upstream algorithm.

The separately vendored geometry snapshot is in
[`deployment/runtime/geometry/pixal3d/`](deployment/runtime/geometry/pixal3d).
Its [snapshot notice](deployment/runtime/geometry/NOTICE.md),
[Microsoft MIT license](deployment/runtime/geometry/TRELLIS2-MIT.txt), and
[Tencent MIT license](deployment/runtime/geometry/TENCENT-MIT.txt) accompany it.

The `timestep_embedding` implementation in
[`supratex/models/sparse_structure_flow.py`](supratex/models/sparse_structure_flow.py)
and its
[vendored geometry counterpart](deployment/runtime/geometry/pixal3d/models/sparse_structure_flow.py)
credits OpenAI's GLIDE. The applicable
[OpenAI GLIDE MIT license](#openai-glide-mit) retains
Copyright (c) 2021 OpenAI.

## FireRed material-model runtime

`supratex/frontend/material_model/runtime/` includes code adapted from
FireRed-Image-Edit and the factorized material-model implementation. Incorporated
FireRed portions remain licensed under Apache License 2.0. Original source
copyright notices are retained, and adaptations for this distribution are marked
in the source files. The project MIT license applies to original SUPRA-Tex
contributions and does not relicense those upstream portions.

The full [FireRed Apache-2.0 license](#firered-apache-2.0) is included,
with the upstream statement: Copyright 2026 FireRed-Image-Edit Authors. All Rights Reserved.

Release adaptations include the local inference entrypoint, explicit external
model/checkpoint paths, input validation, and a self-contained import layout.

## Vendored deployment code

The following source code is present in this repository; it is not merely an
external installation requirement. The recorded snapshot origins and limits of
the available version information are in
[`deployment/runtime/SOURCE_PROVENANCE.json`](deployment/runtime/SOURCE_PROVENANCE.json).
Top-level licenses do not override the terms of incorporated subcomponents.

| Distributed path | Source and applicable notices |
| --- | --- |
| [`deployment/runtime/qwen/`](deployment/runtime/qwen) | Qwen material-model runtime, including `diffsynth`; its top-level [Apache-2.0 license](deployment/runtime/qwen/LICENSE) is retained. The [runtime third-party notices](deployment/runtime/qwen/THIRD_PARTY_NOTICES.md) map MIT subcomponents and document the excluded YuE2 audio VAE and pipeline. |
| [`deployment/runtime/moge/`](deployment/runtime/moge) | Microsoft MoGe source under MIT. The bundled [license file](deployment/runtime/moge/LICENSE) contains both the Microsoft MIT license and the Apache-2.0 text for incorporated DINOv2 code. [`moge/model/dinov2/`](deployment/runtime/moge/moge/model/dinov2) retains Meta copyright and Apache-2.0 headers. |
| [`deployment/runtime/utils3d/`](deployment/runtime/utils3d) | EasternJournalist utils3d; [MIT license](deployment/runtime/utils3d/LICENSE), Copyright (c) 2022 EasternJournalist. |
| [`deployment/runtime/naf/`](deployment/runtime/naf) | Valeo NAF snapshot with its top-level [Apache-2.0 license](deployment/runtime/naf/LICENSE). The component-specific DINOv3, DPIR, and Restormer notices and the REDNet exclusion are listed below. |

### Component-specific source licenses

| Distributed path | Upstream component | Retained license |
| --- | --- | --- |
| [`deployment/runtime/naf/src/layers/rope.py`](deployment/runtime/naf/src/layers/rope.py) | Meta DINOv3, as stated in the source header | [DINOv3 License Agreement](#dinov3-license); this is Meta's custom agreement, not MIT or Apache-2.0. |
| [`deployment/runtime/naf/src/model/ircnn.py`](deployment/runtime/naf/src/model/ircnn.py) | `cszn/DPIR`, `models/network_dncnn.py` | [MIT](#dpir-mit), Copyright (c) 2020 Kai Zhang. |
| [`deployment/runtime/naf/src/model/restormer.py`](deployment/runtime/naf/src/model/restormer.py) | `swz30/Restormer` | [MIT](#restormer-mit), Copyright (c) 2022 Syed Waqas Zamir and contributors. |
| [`deployment/runtime/qwen/diffsynth/models/ace_step_residual_fsq.py`](deployment/runtime/qwen/diffsynth/models/ace_step_residual_fsq.py) | `lucidrains/vector-quantize-pytorch`, `residual_fsq.py` | [MIT](#vector-quantize-pytorch-mit), Copyright (c) 2020 Phil Wang. |

The five additional license texts were retrieved in full from these official
upstream files on 2026-10-14:

- [DINOv3 license](https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md)
  → [DINOv3 License Agreement](#dinov3-license).
- [vector-quantize-pytorch license](https://raw.githubusercontent.com/lucidrains/vector-quantize-pytorch/master/LICENSE)
  → [vector-quantize-pytorch - MIT](#vector-quantize-pytorch-mit).
- [DPIR license](https://raw.githubusercontent.com/cszn/DPIR/master/LICENSE)
  → [DPIR - MIT](#dpir-mit).
- [Restormer license](https://raw.githubusercontent.com/swz30/Restormer/main/LICENSE.md)
  → [Restormer - MIT](#restormer-mit).
- [OpenAI GLIDE license](https://raw.githubusercontent.com/openai/glide-text2im/main/LICENSE)
  → [OpenAI GLIDE - MIT](#openai-glide-mit).

These upstream branch URLs record where the attached license texts were
obtained; they do not establish the exact revisions of the vendored source.

## Excluded source components

The following unrelated functionality is not part of this source release:

- The optional REDNet upsampler (`deployment/runtime/naf/src/model/rednet.py`)
  and its package-level export have been removed. The NAF model used by the
  geometry frontend and its Torch Hub entrypoint remain available.
- The YuE2 audio VAE (`deployment/runtime/qwen/diffsynth/models/yue2_vae.py`),
  its model-registry entry, and the dependent music-generation pipeline
  (`deployment/runtime/qwen/diffsynth/pipelines/yue2.py`) have been removed.
  The Qwen-Image-2.1 image VAE, pipeline, and registry entries used by Supra-Tex
  are separate components and remain available. The
  [Qwen runtime notice](deployment/runtime/qwen/THIRD_PARTY_NOTICES.md)
  records the same release boundary.

These exclusions do not grant permission for the removed code or infer its
terms from the surrounding runtime's Apache-2.0 license.


## Separately installed dependencies and weights

Separately installed packages such as PyTorch, FlexGEMM, CuMesh, O-Voxel,
nvdiffrast, attention extensions, Diffusers, and Transformers retain their own
upstream terms. The vendored sources listed above have their own notices even
when an installed package of the same name is also used.

Model weights, including external Qwen, MoGe-2, NAF, and RMBG-2 weights, have
separate model terms. The MIT license for original SUPRA-Tex source does not
license model weights, adapters, or the base models from which they are derived.
In particular, Apache-2.0 runtime source does not establish the license of
Qwen-Image-2.1 weights or adapters. Follow the license and model card supplied
with each separately distributed weight artifact.

## License texts

<a id="microsoft-mit"></a>

### Microsoft TRELLIS / TRELLIS.2 - MIT

```text
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

<a id="tencent-mit"></a>

### Tencent - MIT

```text
MIT License

Copyright (c) 2026 Tencent.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<a id="openai-glide-mit"></a>

### OpenAI GLIDE - MIT

```text
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<a id="firered-apache-2.0"></a>

### FireRed-Image-Edit - Apache-2.0

```text
                              Apache License
                        Version 2.0, January 2004
                     http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   "Licensor" shall mean the copyright owner or entity authorized by
   the copyright owner that is granting the License.

   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity. For the purposes of this definition,
   "control" means (i) the power, direct or indirect, to cause the
   direction or management of such entity, whether by contract or
   otherwise, or (ii) ownership of fifty percent (50%) or more of the
   outstanding shares, or (iii) beneficial ownership of such entity.

   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.

   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.

   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.

   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License, as indicated by a
   copyright notice that is included in or attached to the work
   (an example is provided in the Appendix below).

   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based on (or derived from) the Work and for which the
   editorial revisions, annotations, elaborations, or other modifications
   represent, as a whole, an original work of authorship. For the purposes
   of this License, Derivative Works shall not include works that remain
   separable from, or merely link (or bind by name) to the interfaces of,
   the Work and Derivative Works thereof.

   "Contribution" shall mean any work of authorship, including
   the original version of the Work and any modifications or additions
   to that Work or Derivative Works thereof, that is intentionally
   submitted to Licensor for inclusion in the Work by the copyright owner
   or by an individual or Legal Entity authorized to submit on behalf of
   the copyright owner. For the purposes of this definition, "submitted"
   means any form of electronic, verbal, or written communication sent
   to the Licensor or its representatives, including but not limited to
   communication on electronic mailing lists, source code control systems,
   and issue tracking systems that are managed by, or on behalf of, the
   Licensor for the purpose of discussing and improving the Work, but
   excluding communication that is conspicuously marked or otherwise
   designated in writing by the copyright owner as "Not a Contribution."

   "Contributor" shall mean Licensor and any individual or Legal Entity
   on behalf of whom a Contribution has been received by Licensor and
   subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   copyright license to reproduce, prepare Derivative Works of,
   publicly display, publicly perform, sublicense, and distribute the
   Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   (except as stated in this section) patent license to make, have made,
   use, offer to sell, sell, import, and otherwise transfer the Work,
   where such license applies only to those patent claims licensable
   by such Contributor that are necessarily infringed by their
   Contribution(s) alone or by combination of their Contribution(s)
   with the Work to which such Contribution(s) was submitted. If You
   institute patent litigation against any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Work
   or a Contribution incorporated within the Work constitutes direct
   or contributory patent infringement, then any patent licenses
   granted to You under this License for that Work shall terminate
   as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
   Work or Derivative Works thereof in any medium, with or without
   modifications, and in Source or Object form, provided that You
   meet the following conditions:

   (a) You must give any other recipients of the Work or
       Derivative Works a copy of this License; and

   (b) You must cause any modified files to carry prominent notices
       stating that You changed the files; and

   (c) You must retain, in the Source form of any Derivative Works
       that You distribute, all copyright, patent, trademark, and
       attribution notices from the Source form of the Work,
       excluding those notices that do not pertain to any part of
       the Derivative Works; and

   (d) If the Work includes a "NOTICE" text file as part of its
       distribution, then any Derivative Works that You distribute must
       include a readable copy of the attribution notices contained
       within such NOTICE file, excluding those notices that do not
       pertain to any part of the Derivative Works, in at least one
       of the following places: within a NOTICE text file distributed
       as part of the Derivative Works; within the Source form or
       documentation, if provided along with the Derivative Works; or,
       within a display generated by the Derivative Works, if and
       wherever such third-party notices normally appear. The contents
       of the NOTICE file are for informational purposes only and
       do not modify the License. You may add Your own attribution
       notices within Derivative Works that You distribute, alongside
       or as an addendum to the NOTICE text from the Work, provided
       that such additional attribution notices cannot be construed
       as modifying the License.

   You may add Your own copyright statement to Your modifications and
   may provide additional or different license terms and conditions
   for use, reproduction, or distribution of Your modifications, or
   for any such Derivative Works as a whole, provided Your use,
   reproduction, and distribution of the Work otherwise complies with
   the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
   names, trademarks, service marks, or product names of the Licensor,
   except as required for reasonable and customary use in describing the
   origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
   agreed to in writing, Licensor provides the Work (and each
   Contributor provides its Contributions) on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied, including, without limitation, any warranties or conditions
   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
   PARTICULAR PURPOSE. You are solely responsible for determining the
   appropriateness of using or redistributing the Work and assume any
   risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
   whether in tort (including negligence), contract, or otherwise,
   unless required by applicable law (such as deliberate and grossly
   negligent acts) or agreed to in writing, shall any Contributor be
   liable to You for damages, including any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of this License or out of the use or inability to use the
   Work (including but not limited to damages for loss of goodwill,
   work stoppage, computer failure or malfunction, or any and all
   other commercial damages or losses), even if such Contributor
   has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
   the Work or Derivative Works thereof, You may choose to offer,
   and charge a fee for, acceptance of support, warranty, indemnity,
   or other liability obligations and/or rights consistent with this
   License. However, in accepting such obligations, You may act only
   on Your own behalf and on Your sole responsibility, not on behalf
   of any other Contributor, and only if You agree to indemnify,
   defend, and hold each Contributor harmless for any liability
   incurred by, or claims asserted against, such Contributor by reason
   of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

   To apply the Apache License to your work, attach the following
   boilerplate notice, with the fields enclosed by brackets "[]"
   replaced with your own identifying information. (Don't include
   the brackets!)  The text should be enclosed in the appropriate
   comment syntax for the file format. We also recommend that a
   file or class name and description of purpose be included on the
   same "printed page" as the copyright notice for easier
   identification within third-party archives.

Copyright 2026 FireRed-Image-Edit Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

<a id="dinov3-license"></a>

### DINOv3 License Agreement

```text
# DINOv3 License

*Last Updated: August 19, 2025*

**“Agreement”** means the terms and conditions for use, reproduction, distribution and modification of the DINO Materials set forth herein.

**“DINO Materials”** means, collectively, Documentation and the models, software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code, and other elements of the foregoing distributed by Meta and made available under this Agreement.

**“Documentation”** means the specifications, manuals and documentation accompanying
DINO Materials distributed by Meta.

**“Licensee”** or **“you”** means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

**“Meta”** or **“we”** means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) or Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).

**“Sanctions”** means any economic or trade sanctions or restrictions administered or enforced by the United States (including the Office of Foreign Assets Control of the U.S. Department of the Treasury (“OFAC”), the U.S. Department of State and the U.S. Department of Commerce), the United Nations, the European Union, or the United Kingdom.

**“Trade Controls”** means any of the following: Sanctions and applicable export and import controls.

By clicking “I Accept” below or by using or distributing any portion or element of the DINO Materials, you agree to be bound by this Agreement.

## 1. License Rights and Redistribution.

a. <ins>Grant of Rights</ins>. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the DINO Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the DINO Materials.

b. <ins>Redistribution and Use</ins>.

i. Distribution of DINO Materials, and any derivative works thereof, are subject to the terms of this Agreement. If you distribute or make the DINO Materials, or any derivative works thereof, available to a third party, you may only do so under the terms of this Agreement and you shall provide a copy of this Agreement with any such DINO Materials.

ii.  If you submit for publication the results of research you perform on, using, or otherwise in connection with DINO Materials, you must acknowledge the use of DINO Materials in your publication.

iii. Your use of the DINO Materials must comply with applicable laws and regulations, including Trade Control Laws and applicable privacy and data protection laws.

iv. Your use of the DINO Materials will not involve or encourage others to reverse engineer, decompile or discover the underlying components of the DINO Materials.

v. You are not the target of Trade Controls and your use of DINO Materials must comply with Trade Controls. You agree not to use, or permit others to use, DINO Materials for any activities subject to the International Traffic in Arms Regulations (ITAR) or end uses prohibited by Trade Controls, including those related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons.

## 2. User Support.

Your use of the DINO Materials is done at your own discretion; Meta does not process any information nor provide any service in relation to such use.  Meta is under no obligation to provide any support services for the DINO Materials. Any support provided is “as is”, “with all faults”, and without warranty of any kind.

## 3. Disclaimer of Warranty.

UNLESS REQUIRED BY APPLICABLE LAW, THE DINO MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE DINO MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE DINO MATERIALS AND ANY OUTPUT AND RESULTS.

## 4. Limitation of Liability.

IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY DIRECT OR INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

## 5. Intellectual Property.

a. Subject to Meta’s ownership of DINO Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the DINO Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.

b. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the DINO Materials, outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the DINO Materials.

## 6. Term and Termination.

The term of this Agreement will commence upon your acceptance of this Agreement or access to the DINO Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the DINO Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.

## 7. Governing Law and Jurisdiction.

This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.

## 8. Modifications and Amendments.

Meta may modify this Agreement from time to time; provided that they are similar in spirit to the current version of the Agreement, but may differ in detail to address new problems or concerns. All such changes will be effective immediately. Your continued use of the DINO Materials after any modification to this Agreement constitutes your agreement to such modification. Except as provided in this Agreement, no modification or addition to any provision of this Agreement will be binding unless it is in writing and signed by an authorized representative of both you and Meta.
```

<a id="dpir-mit"></a>

### DPIR - MIT

```text
MIT License

Copyright (c) 2020 Kai Zhang (cskaizhang@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

<a id="restormer-mit"></a>

### Restormer - MIT

```text
MIT License

Copyright (c) 2022 Syed Waqas Zamir and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

<a id="vector-quantize-pytorch-mit"></a>

### vector-quantize-pytorch - MIT

```text
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<a id="qwen-research-license"></a>

### Qwen-Image-2.1 Research License - material adapter and external base weights

```text
Qwen RESEARCH LICENSE AGREEMENT

Qwen RESEARCH LICENSE AGREEMENT Release Date: September 20, 2026

By clicking to agree or by using or distributing any portion or element of the Qwen Materials, you will be deemed to have recognized and accepted the content of this Agreement, which is effective immediately.

1. Definitions
    a. This Qwen RESEARCH LICENSE AGREEMENT (this "Agreement") shall mean the terms and conditions for use, reproduction, distribution and modification of the Materials as defined by this Agreement.
    b. "We" (or "Us") shall mean Hangzhou Tongyi Laboratory Technology Co., Ltd.
    c. "You" (or "Your") shall mean a natural person or legal entity exercising the rights granted by this Agreement and/or using the Materials for any purpose and in any field of use.
    d. "Third Parties" shall mean individuals or legal entities that are not under common control with us or you.
    e. "Qwen" shall mean the large language models, diffusion models, and software and algorithms, consisting of trained model weights, parameters (including optimizer states), machine-learning model code, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing distributed by us.
    f. "Materials" shall mean, collectively, our proprietary Qwen and Documentation (and any portion thereof) made available under this Agreement.
    g. "Source" form shall mean the preferred form for making modifications, including but not limited to model source code, documentation source, and configuration files.
    h. "Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.
    i. "Non-Commercial" shall mean for research or evaluation purposes only.

2. Grant of Rights
    a. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under our intellectual property or other rights owned by us embodied in the Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Materials FOR NON-COMMERCIAL PURPOSES ONLY.
    b. You shall not use the Materials for any commercial purpose without obtaining a separate commercial license from us. If you wish to use the Materials commercially, you shall request a license from us at model-business@notice.qwencloud.com.

3. Redistribution
Subject to Section 2 (Grant of Rights), you may distribute copies or make the Materials, or derivative works thereof, available as part of a product or service that contains any of them, with or without modifications, and in Source or Object form, provided that you meet the following conditions:
    a. You shall give any other recipients of the Materials or derivative works a copy of this Agreement;
    b. You shall cause any modified files to carry prominent notices stating that you changed the files;
    c. You shall retain in all copies of the Materials that you distribute the following attribution notices within a "Notice" text file distributed as a part of such copies: "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved."; and
    d. You may add your own copyright statement to your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of your modifications, or for any such derivative works as a whole, provided your use, reproduction, and distribution of the work otherwise complies with the terms and conditions of this Agreement.

4. Rules of use
    a. The Materials may be subject to export controls or restrictions in China, the United States or other countries or regions. You shall comply with applicable laws and regulations in your use of the Materials.
    b. If you use the Materials or any outputs or results therefrom to create, train, fine-tune, or improve an AI model that is distributed or made available, you shall prominently display “Built with Qwen” or “Improved using Qwen” in the related product documentation.
    c. You shall not use "Qwen" as the primary name or identifier of any derivative works or products; reasonable descriptive use (e.g., "fine-tuned from Qwen Image") is permitted.

5. Intellectual Property
    a. We retain ownership of all intellectual property rights in and to the Materials and derivatives made by or for us. Conditioned upon compliance with the terms and conditions of this Agreement, with respect to any derivative works and modifications of the Materials that are made by you, you are and will be the owner of such derivative works and modifications.
    b. No trademark license is granted to use the trade names, trademarks, service marks, or product names of us, except as required to fulfill notice requirements under this Agreement or as required for reasonable and customary use in describing and redistributing the Materials.
    c. If you commence a lawsuit or other proceedings (including a cross-claim or counterclaim in a lawsuit) against us or any entity alleging that the Materials or any output therefrom, or any part of the foregoing, infringe any intellectual property or other right owned or licensable by you, then all licenses granted to you under this Agreement shall terminate as of the date such lawsuit or other proceeding is commenced or brought.

6. Disclaimer of Warranty and Limitation of Liability
    a. We are not obligated to support, update, provide training for, or develop any further version of the Qwen Materials or to grant any license thereto.
    b. THE MATERIALS ARE PROVIDED "AS IS" WITHOUT ANY EXPRESS OR IMPLIED WARRANTY OF ANY KIND INCLUDING WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR FITNESS FOR A PARTICULAR PURPOSE. WE MAKE NO WARRANTY AND ASSUME NO RESPONSIBILITY FOR THE SAFETY OR STABILITY OF THE MATERIALS AND ANY OUTPUT THEREFROM.
    c. IN NO EVENT SHALL WE BE LIABLE TO YOU FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO ANY DIRECT, OR INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM YOUR USE OR INABILITY TO USE THE MATERIALS OR ANY OUTPUT OF IT, NO MATTER HOW IT’S CAUSED.
    d. You will defend, indemnify and hold harmless us from and against any claim by any third party arising out of or related to your use or distribution of the Materials.

7. Survival and Termination.
    a. The term of this Agreement shall commence upon your acceptance of this Agreement or access to the Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein.
    b. We may terminate this Agreement if you breach any of the terms or conditions of this Agreement. Upon termination of this Agreement, you must delete and cease use of the Materials. Sections 6 and 8 shall survive the termination of this Agreement.

8. Governing Law and Jurisdiction.
    a. This Agreement and any dispute arising out of or relating to it will be governed by the laws of China, without regard to conflict of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement.
    b. The People's Courts in Hangzhou City shall have exclusive jurisdiction over any dispute arising out of this Agreement.

9. Other Terms and Conditions.
    a. Any arrangements, understandings, or agreements regarding the Material not stated herein are separate from and independent of the terms and conditions of this Agreement. You shall request a separate license from us, if you use the Materials in ways not expressly agreed to in this Agreement.
    b. We shall not be bound by any additional or different terms or conditions communicated by you unless expressly agreed.
```
