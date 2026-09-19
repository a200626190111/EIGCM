# Notice and attribution

## Repository scope

This repository contains a complete Radiance 6.1 source tree together with
EIGCM additions and modifications. It is therefore a derived Radiance source
distribution, not a standalone collection containing only EIGCM files.

The upstream project is Radiance, a validated lighting simulation system:

- Official project: https://www.radiance-online.org/
- Upstream source mirror: https://github.com/LBNL-ETA/Radiance

This repository is independently maintained and is not an official Radiance or
Lawrence Berkeley National Laboratory distribution. The names of the upstream
institutions and contributors are not used to endorse EIGCM.

## Upstream copyright and license

Radiance 6.1 is Copyright (c) 1990-2025, The Regents of the University of
California, through Lawrence Berkeley National Laboratory. The complete
Radiance Software License, Version 2.0, is retained in `License.txt`.

The upstream source, copyright notices, license conditions, and disclaimer are
retained in this redistribution. Binary redistributions must also reproduce
the copyright notice, license conditions, and disclaimer in their accompanying
documentation or materials, as required by `License.txt`.

## EIGCM enhancements

The EIGCM work in this repository includes:

- `eigcm_scheduler.py` and its configuration template;
- the EIGCM glare, direct-lobe, specular, TensorTree, and BSDF utilities;
- modifications to Radiance matrix and ray-tracing support used by EIGCM;
- EIGCM documentation, tests, and the four manuscript validation cases.

Unless a file contains a more specific notice, these publicly released EIGCM
enhancements are distributed under the Radiance Software License, Version 2.0.
Manuscript drafts, benchmark outputs, rendered HDR images, and experimental GPU
implementations from the development tree are not included.

## File-specific notices

The Radiance source distribution contains files and incorporated components
with their own copyright or license notices, including `evalglare` and several
utility libraries. Those notices have been retained in the relevant source
files and continue to apply to those files. Nothing in this notice replaces or
limits a file-specific copyright or license statement.
