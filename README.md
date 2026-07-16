# MNZI
[Build](https://github.com/thepaulbuchanan/MNZI/actions/workflows/build.yml/badge.svg)

Lean 4 formalisation of the **Motivic Neural Zeta Integrator (MNZI)** programme:
a series of papers on the Riemann zeta function, Kreĭn-space operator theory,
and the golden ratio.

Every result marked **PROVED** below is machine-verified in Lean 4 against
Mathlib, contains no `sorry`, and depends only on Lean's standard axioms
(`propext`, `Classical.choice`, `Quot.sound`).

MNZI is one component of the broader **Reflective Shade Duality (RSD)**
programme; the others (Local Index Theory, Dual Quantum Gravity) are not
formalised and are not in this repository.

---

## Status
[Build](https://github.com/thepaulbuchanan/MNZI/actions/workflows/build.yml/badge.svg)


| | |
|---|---|
| Papers | 38 |
| Lean files | 34 <!-- TODO: confirm after migration --> |
| Theorems | ~630 <!-- TODO: confirm; cheatsheet figure --> |
| `sorry` | none |
| Non-standard axioms | none |
| Toolchain | `leanprover/lean4:v4.28.0` |

---

## Verifying

You need Lean 4 (via [elan](https://github.com/leanprover/elan)). Everything
else is fetched automatically.

```bash
git clone https://github.com/thepaulbuchanan/MNZI
cd MNZI
lake exe cache get     # downloads prebuilt Mathlib — do not skip
lake build             # builds all files
```

To check a single paper rather than the whole series:

```bash
lake build MNZI.KreinAPSFormula     # Paper C only, plus its imports
```

To confirm the axiom footprint of any result, add to the file:

```lean
#print axioms MNZI.buchanan_aps_formula
```

Expected output: `[propext, Classical.choice, Quot.sound]`.

---

## Layout

```
MNZI/          all Lean source, one file per paper
papers/        per-paper README, PDF, figures
lakefile.toml  build configuration
```

There is one copy of each Lean file. `papers/X/README.md` links to it rather
than duplicating it.

---

## Papers

<!-- TODO: complete this table. Columns: paper, title, Lean file, DOI.
     Use the CONCEPT DOI (Zenodo's "cite all versions"), not a version DOI.
     Lean filenames must match what is actually in MNZI/ after migration —
     several differ from earlier drafts. -->

| Paper | Title | Lean | DOI |
|---|---|---|---|
| A | The GUE Mode Theorem | [`GoldenRatioGUEMode.lean`](MNZI/GoldenRatioGUEMode.lean) | [10.5281/zenodo.19160183](https://doi.org/10.5281/zenodo.19160183) |
| C | Kreĭn APS Formula | [`KreinAPSFormula.lean`](MNZI/KreinAPSFormula.lean) | [10.5281/zenodo.19160725](https://doi.org/10.5281/zenodo.19160725) |
| … | | | |

---

## Structure of the formalisation

`MNZI/Foundations.lean` holds the canonical definitions shared across the
series — the coil invariant, the golden ratio, the gap constants, the
functional-equation involution, the Kreĭn structures, the derivative tower.
Every paper file imports it. Nothing in Foundations depends on any paper.

`MNZI/CrownJewels.lean` holds the named aliases for the programme's principal
results. It imports Foundations and the papers; nothing imports it.

<!-- TODO: this section describes the target state. Update once migration
     from the per-paper Core is complete. -->

---

## Scope and honesty

The programme makes no claim to prove the Riemann Hypothesis.

Results are labelled by verification status, following the convention used
throughout the papers:

| Label | Meaning |
|---|---|
| **PROVED** | Proved in full, sorry-free, standard axioms only. |
| **PROVED†** | Proved in full, but carrying a property absent from Mathlib as an explicit named hypothesis, documented in the docstring. |
| **STATEMENT** | A faithful `def` over the actual objects, with a comment recording what infrastructure is missing. No `sorry`. |
| **PENDING** | A `def` stating a proposition. No proof claimed. No `sorry`. |

Conjectures and open questions are stated as `Prop`-valued definitions, never
patched with `sorry`.

Where formalisation has contradicted a manuscript, the contradiction is
recorded in the Lean file rather than silently corrected. Known instances are
listed in the relevant paper's README.

---

## Mathlib gaps encountered

Formalising this programme surfaced results that appear to be absent from
Mathlib. Listed as observations.

- **Schwarz reflection for the completed zeta.**
  `completedRiemannZeta (conj s) = conj (completedRiemannZeta s)`.
  Carried as an explicit hypothesis (`hconj`) in several files. Its absence is
  the reason those results are **PROVED†** rather than **PROVED**.

<!-- TODO: add further gaps as they are confirmed. -->

---

## Citing

<!-- TODO: add the code DOI once the first GitHub Release is archived by
     Zenodo. Note this is separate from the per-paper DOIs above. -->

For an individual paper, cite its DOI in the table above. For the
formalisation as a whole, cite the repository.

---

## Licence

Apache 2.0. See [LICENSE](LICENSE).
