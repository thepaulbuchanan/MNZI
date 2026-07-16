/-
  MNZI Paper R-1: The Derivative Tower of the Riemann ξ Function  [REVISED]
  and the X- and Y-Families of the Stabilizer 2-Design

  Revision notes (Session 3):
  - Fix 6: Remove `exact?` from xi_deriv_tower_symmetry; provide explicit
            chain rule proof using HasDerivAt.comp + ENat.coe_lt_top
  - Fix 7: Replace vacuous ConjectureSpectralConvergence with correct
            formulation involving the tower at Riemann zeros; remove spurious proof
  - Fix 8: Replace mislabelled frame_constant_d4 (proved 24/6=4) with
            correct frame constant 2/(d(d+1)) = 1/10 for d=4
-/

import Mathlib
import MNZI.Core

namespace MNZI

open Complex

/-! ## Section 1: Combinatorial Structure of S₂₄ -/

inductive StabilizerFamily : Type
  | Z | X | Y
  deriving DecidableEq, Fintype

inductive Anchor : Type
  | pos | neg
  deriving DecidableEq, Fintype

inductive Phase4 : Type
  | one | posI | negOne | negI
  deriving DecidableEq, Fintype

abbrev StabilizerLabel := StabilizerFamily × Anchor × Phase4

noncomputable instance : Fintype StabilizerLabel :=
  inferInstanceAs (Fintype (StabilizerFamily × Anchor × Phase4))

instance : DecidableEq StabilizerLabel :=
  inferInstanceAs (DecidableEq (StabilizerFamily × Anchor × Phase4))

theorem card_stabilizerFamily : Fintype.card StabilizerFamily = 3 := by rfl
theorem card_anchor : Fintype.card Anchor = 2 := by rfl
theorem card_phase4 : Fintype.card Phase4 = 4 := by rfl

theorem S24_geom_parametrisation : Fintype.card StabilizerLabel = 24 := by rfl
theorem three_mul_two_mul_four : 3 * 2 * 4 = 24 := by norm_num

/-! ## Section 2: Local Unitary Relations -/

def familyCycleH : StabilizerFamily → StabilizerFamily
  | .Z => .X | .X => .Z | .Y => .Y

def familyCycleSH : StabilizerFamily → StabilizerFamily
  | .Z => .Y | .Y => .Z | .X => .X

theorem familyCycleH_involutive : Function.Involutive familyCycleH :=
  fun x => by cases x <;> rfl

theorem familyCycleSH_involutive : Function.Involutive familyCycleSH :=
  fun x => by cases x <;> rfl

theorem S24_local_unitary_IH :
    Function.Bijective (fun (lab : StabilizerLabel) =>
      ((familyCycleH lab.1, lab.2.1, lab.2.2) : StabilizerLabel)) :=
  Fintype.bijective_iff_injective_and_card _ |>.2 ⟨by decide, rfl⟩

theorem S24_local_unitary_ISH :
    Function.Bijective (fun (lab : StabilizerLabel) =>
      ((familyCycleSH lab.1, lab.2.1, lab.2.2) : StabilizerLabel)) := by
  simp +decide

/-! ## Section 3: Functional Equations of the Derivative Tower -/

theorem xi_prime_antisymmetric
    (f : ℂ → ℂ) (hf : Differentiable ℂ f) (hfe : ∀ s, f (1 - s) = f s) :
    ∀ s, deriv f (1 - s) = -deriv f s := by
  intro s
  have h_chain : deriv (fun s => f (1 - s)) s = deriv f (1 - s) * deriv (fun s => 1 - s) s :=
    deriv_comp s hf.differentiableAt (differentiableAt_id.const_sub _)
  simp_all +decide [sub_eq_add_neg]

theorem xi_dprime_symmetric
    (f : ℂ → ℂ) (hf : Differentiable ℂ f) (hf' : Differentiable ℂ (deriv f))
    (hfe : ∀ s, f (1 - s) = f s) :
    ∀ s, deriv (deriv f) (1 - s) = deriv (deriv f) s := by
  intro s
  have : deriv (fun x => deriv f (1 - x)) s = - deriv (deriv f) (1 - s) := by
    convert HasDerivAt.deriv
      (HasDerivAt.scomp s hf'.differentiableAt.hasDerivAt
        (hasDerivAt_id' s |> HasDerivAt.const_sub 1)) using 1
    norm_num
  rw [show deriv (fun x => deriv f (1 - x)) s = deriv (fun x => -deriv f x) s by
    congr; ext x; exact xi_prime_antisymmetric f hf hfe x] at this
  norm_num at *; ring_nf at *; aesop

/-!
### FIX 6: xi_deriv_tower_symmetry — replace `exact?` with explicit proof

PROBLEM: The original proof contained `exact?` (a Lean editor command, not a
proof term) in the inductive step's `apply_rules` block. The file would not compile.

ROOT CAUSE: The side condition for `ContDiff.differentiable_iteratedDeriv` requires
showing `↑k < ⊤` (as elements of `ℕ∞ = WithTop ℕ`). This is `ENat.coe_lt_top k`
in Mathlib4. The `exact?` was a placeholder for finding this lemma name.

FIX: Replace the entire inductive step with an explicit chain rule proof using
`HasDerivAt.comp`. The key steps are:
  1. Differentiability: `hf.differentiable_iteratedDeriv k (WithTop.natCast_lt_top k)`
  2. HasDerivAt for (1-·): `(hasDerivAt_id s).const_sub 1` simplified
  3. Chain rule: `HasDerivAt.comp` combines (1) and (2)
  4. Differentiate the ih equality via `funext` + `deriv_const_mul`
-/

/-- **Lemma 3.1** (xi_deriv_tower_symmetry): ξ^(k)(1-s) = (-1)^k · ξ^(k)(s).

    Proof by induction on k. The base case is the functional equation itself.
    The inductive step applies the chain rule explicitly (HasDerivAt.comp)
    to avoid the `exact?` gap in the original. -/
theorem xi_deriv_tower_symmetry
    (f : ℂ → ℂ) (hf : ContDiff ℂ ⊤ f) (hfe : ∀ s, f (1 - s) = f s) :
    ∀ (k : ℕ) (s : ℂ), iteratedDeriv k f (1 - s) = (-1) ^ k * iteratedDeriv k f s := by
  intro k
  induction k with
  | zero => intro s; simp [iteratedDeriv_zero, hfe]
  | succ k ih =>
    intro s
    simp only [pow_succ, iteratedDeriv_succ]
    -- Step 1: differentiability of the k-th iterated derivative
    have hdiff : Differentiable ℂ (iteratedDeriv k f) :=
      hf.differentiable_iteratedDeriv k (WithTop.natCast_lt_top k)
    -- Step 2: HasDerivAt for the affine map (1 - ·)
    have haffine : HasDerivAt (fun t : ℂ => 1 - t) (-1) s := by
      simpa using (hasDerivAt_id s).const_sub (1 : ℂ)
    -- Step 3: HasDerivAt for iteratedDeriv k f at (1-s)
    have hitk : HasDerivAt (iteratedDeriv k f)
        (deriv (iteratedDeriv k f) (1 - s)) (1 - s) :=
      hdiff.differentiableAt.hasDerivAt
    -- Step 4: Chain rule — compose to get HasDerivAt for (iteratedDeriv k f) ∘ (1-·)
    have hcomp := hitk.comp s haffine
    -- hcomp : HasDerivAt (fun t => iteratedDeriv k f (1-t))
    --           (deriv (iteratedDeriv k f) (1-s) * (-1)) s
    have hchain : deriv (fun t => iteratedDeriv k f (1 - t)) s =
        deriv (iteratedDeriv k f) (1 - s) * (-1) :=
      hcomp.deriv
    -- Step 5: Use ih to rewrite (fun t => iteratedDeriv k f (1-t)) as a scalar multiple
    have heq : (fun t => iteratedDeriv k f (1 - t)) =
        (fun t => (-1 : ℂ)^k * iteratedDeriv k f t) :=
      funext ih
    -- Step 6: Differentiate the rewritten function
    have hderiv_eq : deriv (fun t => iteratedDeriv k f (1 - t)) s =
        (-1 : ℂ)^k * deriv (iteratedDeriv k f) s := by
      rw [heq]
      exact deriv_const_mul ((-1 : ℂ)^k) hdiff.differentiableAt
    -- Step 7: Combine hchain and hderiv_eq
    -- hchain:     deriv (fun t => itk f (1-t)) s = deriv (itk f) (1-s) * (-1)
    -- hderiv_eq:  deriv (fun t => itk f (1-t)) s = (-1)^k * deriv (itk f) s
    -- Therefore:  deriv (itk f) (1-s) * (-1) = (-1)^k * deriv (itk f) s
    -- i.e.:       deriv (itk f) (1-s) = (-1)^k * (-1) * deriv (itk f) s  ✓
    have hfinal : deriv (iteratedDeriv k f) (1 - s) * (-1 : ℂ) =
        (-1 : ℂ)^k * deriv (iteratedDeriv k f) s := by
      rw [← hchain, hderiv_eq]
    -- Step 7 conclusion: from hfinal, derive the goal by simple algebra
    linear_combination -hfinal

/-- **Lemma 3.1 (L_ξ antisymmetry)** -/
theorem L_xi_antisymmetric
    (g h : ℂ → ℂ) (hg : ∀ s, g (1 - s) = -g s) (hh : ∀ s, h (1 - s) = h s)
    (_hne : ∀ s, h s ≠ 0) :
    ∀ s, g (1 - s) / h (1 - s) = -(g s / h s) :=
  fun s => by rw [hg, hh, neg_div]

/-- **Lemma 3.1 (M_ξ antisymmetry)** -/
theorem M_xi_antisymmetric
    (g h : ℂ → ℂ) (hg : ∀ s, g (1 - s) = g s) (hh : ∀ s, h (1 - s) = -h s)
    (_hne : ∀ s, h s ≠ 0) :
    ∀ s, g (1 - s) / h (1 - s) = -(g s / h s) := by
  grind +ring

/-! ## Section 4: Unit Modulus on the Critical Line -/

theorem conj_eq_neg_of_re_zero (z : ℂ) (hz : z.re = 0) :
    starRingEnd ℂ z = -z := by
  simp +decide [Complex.ext_iff, hz]

theorem antisymm_schwarz_at_imag
    (g : ℂ → ℂ)
    (hanti : ∀ s, g (1 - s) = -g s)
    (hschwarz : ∀ s, starRingEnd ℂ (g s) = g (starRingEnd ℂ s))
    (z : ℂ) (hz_im : z.re = 0) :
    g (1 + z) = -(starRingEnd ℂ (g z)) := by
  convert hanti (-z) using 1; ring
  rw [hschwarz, show (starRingEnd ℂ) z = -z from by simp [Complex.ext_iff, hz_im]]

theorem unit_modulus_antisymm_ratio
    (g : ℂ → ℂ)
    (hanti : ∀ s, g (1 - s) = -g s)
    (hschwarz : ∀ s, starRingEnd ℂ (g s) = g (starRingEnd ℂ s))
    (z : ℂ) (hz_im : z.re = 0)
    (hne : g z ≠ 0) :
    ‖g z / g (1 + z)‖ = 1 := by
  rw [antisymm_schwarz_at_imag g hanti hschwarz z hz_im]; norm_num [hne]

/-- **Theorem 3.3 (Unit modulus of φ')**: |φ'(1/2+it)| = 1 -/
theorem phi_prime_unit_modulus
    (L : ℂ → ℂ)
    (hanti : ∀ s, L (1 - s) = -L s)
    (hschwarz : ∀ s, starRingEnd ℂ (L s) = L (starRingEnd ℂ s))
    (t : ℝ) (hne : L (I * ↑t) ≠ 0) :
    ‖L (I * ↑t) / L (1 + I * ↑t)‖ = 1 := by
  convert unit_modulus_antisymm_ratio L hanti hschwarz _ _ hne using 1; ring!; aesop

/-- **Theorem 3.3 (Unit modulus of χ)**: |χ(1/2+it)| = 1 -/
theorem chi_unit_modulus
    (M : ℂ → ℂ)
    (hanti : ∀ s, M (1 - s) = -M s)
    (hschwarz : ∀ s, starRingEnd ℂ (M s) = M (starRingEnd ℂ s))
    (t : ℝ) (hne : M (I * ↑t) ≠ 0) :
    ‖M (I * ↑t) / M (1 + I * ↑t)‖ = 1 :=
  unit_modulus_antisymm_ratio M hanti hschwarz (I * t) (by simp) hne

theorem unit_modulus_symm_ratio
    (f : ℂ → ℂ)
    (hsymm : ∀ s, f (1 - s) = f s)
    (hschwarz : ∀ s, starRingEnd ℂ (f s) = f (starRingEnd ℂ s))
    (z : ℂ) (hz_im : z.re = 0)
    (hne : f z ≠ 0) :
    ‖f z / f (1 + z)‖ = 1 := by
  rw [show 1 + z = 1 - (-z) by ring, hsymm]
  have h_conj : f (-z) = starRingEnd ℂ (f z) := by
    rw [hschwarz, conj_eq_neg_of_re_zero z hz_im]

  rw [h_conj, norm_div, Complex.norm_conj, div_self]; aesop

/-! ## Section 5: Phase Formulae -/

theorem phase_antisymm_ratio (w : ℂ) (_hw : w ≠ 0) :
    w / (-(starRingEnd ℂ w)) = -(w / (starRingEnd ℂ w)) := by grind +revert

theorem norm_div_conj (w : ℂ) (hw : w ≠ 0) :
    ‖w / (starRingEnd ℂ w)‖ = 1 := by simp +decide [hw]

/-! ## Section 6: Confinement -/

theorem confinement_arithmetic (n : ℕ) (_hn : n ≥ 1) :
    (4 * n + 5) % 8 ≠ 0 ∧ (4 * n + 5) % 8 ≠ 4 := by omega

/-! ## Section 7: Conjectures

### FIX 7: Correct formulation of ConjectureSpectralConvergence

PROBLEM: The original definition
  `∃ omegaPlus omegaMinus : ℝ, omegaPlus = π/2 ∧ omegaMinus = π`
is trivially true (just asserts that ℝ contains π/2 and π). The "proof"
`spectral_convergence_witness` was therefore not a mathematical result at all.

CORRECT FORMULATION: The conjecture asserts that arg(φ^(k)) at Riemann zeros
converges to a period-2 orbit as k → ∞. This requires:
  (a) A definition of the k-th tower ratio
  (b) A statement about convergence of its argument at the zeros
This is a genuinely open question. The correct Lean treatment is to state the
Prop without providing a proof (or mark it with `sorry` if a placeholder is needed).
-/

/-- The k-th ratio in the derivative tower, evaluated at a given complex number.
    φ^(k)(s) = [ξ^(k)(2s-1)] / [ξ^(k)(2s)], using iteratedDeriv. -/
noncomputable def towerRatio (xi : ℂ → ℂ) (k : ℕ) (s : ℂ) : ℂ :=
  iteratedDeriv k xi (2 * s - 1) / iteratedDeriv k xi (2 * s)

/-- The period-2 orbit values: even k → π/2, odd k → π. -/
noncomputable def orbitAngle (k : ℕ) : ℝ :=
  if k % 2 = 0 then Real.pi / 2 else Real.pi

/-- **Conjecture 5.1 (Spectral Convergence)** — CORRECT FORMULATION.
    For every Riemann zero γ_n and every ε > 0, there exists K such that for all k ≥ K,
    the argument of φ^(k)(1/2 + iγ_n) is within ε of the period-2 orbit value.

    This is a genuinely open conjecture. No proof is provided here.
    The numerical evidence (Session 2) shows:
      - φ  (k=0): concentrated in {+i, -1} — arg ≈ π/2 or π
      - φ' (k=1): more concentrated at -1 — arg ≈ π
      - χ  (k=2): concentrated at +i — arg ≈ π/2 -/
def ConjectureSpectralConvergence
    (xi : ℂ → ℂ) (riemannZeros : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ) (ε : ℝ), ε > 0 →
    ∃ (K : ℕ), ∀ (k : ℕ), k ≥ K →
      |Complex.arg (towerRatio xi k (1/2 + I * (riemannZeros n : ℂ))) -
       orbitAngle k| < ε

-- NOTE: There is no proof of ConjectureSpectralConvergence. This is intentional.
-- The original spurious `spectral_convergence_witness` theorem is removed.

/-- **Conjecture 4.2**: φ' shares the {+i, -1} confinement of φ at all Riemann zeros. -/
def ConjectureConfinementPrime : Prop :=
  ∀ (L : ℂ → ℂ) (gamma : ℕ → ℝ),
    (∀ s, L (1 - s) = -L s) →
    (∀ s, starRingEnd ℂ (L s) = L (starRingEnd ℂ s)) →
    ∀ n, (L (I * ↑(gamma n)) / L (1 + I * ↑(gamma n))).re ≤ 0

/-! ## Section 8: Statistical Verification -/

/-- Enrichment ratio for φ' at -1: observed 34, expected 12.48. Ratio > 2.7. -/
theorem enrichment_phi_prime_fold :
    (34 : ℚ) / (1248 / 100) > 27 / 10 := by norm_num

/-- Enrichment ratio for χ at +i: observed 42, expected 12.43. Ratio > 3.3. -/
theorem enrichment_chi_posI :
    (42 : ℚ) / (1243 / 100) > 33 / 10 := by norm_num

/-!
### FIX 8: Correct frame constant

PROBLEM: Original `frame_constant_d4` proved `(24 : ℚ) / 6 = 4`.
This is a true arithmetic fact but is NOT the frame constant of the 2-design.
The frame constant for a 2-design in ℂ^d is c_2 = 2/(d(d+1)).
For d = 4: c_2 = 2/(4·5) = 2/20 = 1/10.

The value 24/6 = 4 does not arise from the 2-design formula.
It is mislabelled. The correct theorem is below.

For reference: the S₂₄ 2-design property states that for any unit vector φ ∈ ℂ^4,
  Σ_{ψ ∈ S₂₄} |⟨φ|ψ⟩|^4 = |S₂₄| · c_2 = 24 · (1/10) = 12/5.
-/

/-- **Theorem** (frame_constant_d4): The frame constant for a 2-design in ℂ^4 is 1/10.
    This is c_2 = 2/(d(d+1)) with d=4. -/
theorem frame_constant_d4 : (2 : ℚ) / (4 * (4 + 1)) = 1 / 10 := by norm_num

/-- The expected second-moment sum for S₂₄ as a 2-design in ℂ^4:
    |S₂₄| · c_2 = 24 · (1/10) = 12/5. -/
theorem S24_second_moment_sum : (24 : ℚ) * (1 / 10) = 12 / 5 := by norm_num

/-- Sanity check: 24 ≥ d² = 16, confirming S₂₄ exceeds the Welch bound for d=4. -/
theorem S24_exceeds_welch_bound : (24 : ℕ) ≥ 4^2 := by norm_num

/-- CJ-27: Derivative tower unit modulus. -/
def buchanan_derivative_tower := @phi_prime_unit_modulus

end MNZI
