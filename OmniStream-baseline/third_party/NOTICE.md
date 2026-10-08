# Third-party source attribution

`crowd_counting/tmtb_head.py` preserves the CountingHead source from
`taste_more_taste_better/model/counting_head.py`.

Upstream: https://github.com/syhien/taste_more_taste_better
Paper: Taste More, Taste Better: Diverse Data and Strong Model Boost
Semi-Supervised Crowd Counting, CVPR 2025.

The original OpenMMLab copyright header is retained. The repository's Apache-2.0
license is included as `TMTB_LICENSE`. The model wrapper and training utilities
are new integration code; do not attribute them to the TMTB authors.

OmniStream source in `model/` is preserved from https://github.com/Go2Heart/OmniStream.
The upstream MIT license is retained at the repository root.

Bundled `model/dinov3/` code retains Meta copyright headers and is subject to the
DINOv3 License Agreement included as `DINOV3_LICENSE.md`, retrieved from
https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md (2026-10-08).
Other Meta/OpenMMLab files marked Apache-2.0 retain their original headers;
the Apache-2.0 license text is available in `TMTB_LICENSE`.
The root MIT license does not replace these third-party terms.
