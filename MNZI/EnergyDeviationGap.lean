import Mathlib
import MNZI.Core

/-!
# Energy Deviation Gap — MNZI Paper L

Formalization of the energy deviation functional and gap strip analysis
for the completed Riemann zeta function (xi function).

## Convention
Mathlib uses `Real.log 0 = 0` (not -∞). The zero-exclusion theorems
include the hypothesis `‖ξ(1/2 + it)‖ ≥ 1` to recover classical behavior.

The functional equation `ξ(1-s) = ξ(s)` is from Mathlib
(`completedRiemannZeta_one_sub`). Conjugation symmetry
`ξ(conj s) = conj(ξ(s))` is taken as a hypothesis where needed
(it holds for the completed Riemann zeta but is not yet in Mathlib).

`completedRiemannZeta` is differentiable at all s with s ≠ 0 and s ≠ 1
(from `HurwitzZeta.differentiableAt_completedHurwitzZetaEven`).
-/

open Complex Real

noncomputable section

namespace MNZI

-- ============================================================
-- Section 1: Gap Constants
-- ============================================================

-- `gapHi` and `gapWidth` are imported from `MNZI.Core`.

/-- The gap lower boundary 1 - σ₀ = 13/30. -/
def gapLo : ℝ := 13 / 30

theorem gap_width_formula : gapHi - 1 / 2 = gapWidth := by
  simp [gapHi, gapWidth]; ring

theorem gapLo_eq_one_sub_gapHi : gapLo = 1 - gapHi := by
  simp [gapLo, gapHi]; ring

theorem gapLo_lt_half : gapLo < 1 / 2 := by norm_num [gapLo]

theorem gapLo_gt_zero : gapLo > 0 := by norm_num [gapLo]

-- ============================================================
-- Section 2: Xi function and functional equation
-- ============================================================

/-- Abbreviation for the completed Riemann zeta (xi) function. -/
abbrev xi (s : ℂ) : ℂ := completedRiemannZeta s

/-- The functional equation: ξ(1 - s) = ξ(s). -/
theorem xi_functional_eq (s : ℂ) : xi (1 - s) = xi s :=
  completedRiemannZeta_one_sub s

/-- `completedRiemannZeta` is differentiable at s when s ≠ 0 and s ≠ 1. -/
theorem xi_differentiableAt {s : ℂ} (h0 : s ≠ 0) (h1 : s ≠ 1) :
    DifferentiableAt ℂ xi s :=
  HurwitzZeta.differentiableAt_completedHurwitzZetaEven 0 (Or.inl h0) h1

-- ============================================================
-- Section 3: Energy Deviation Functional
-- ============================================================

/-- The energy deviation functional ΔE(σ,t) = log‖ξ(σ+it)‖ - log‖ξ(1/2+it)‖. -/
def energyDeviation (σ t : ℝ) : ℝ :=
  Real.log ‖xi (↑σ + ↑t * I)‖ - Real.log ‖xi (1 / 2 + ↑t * I)‖

/-- ΔE(1/2, t) = 0 for all t (equilibrium on the critical line). -/
theorem energyDeviation_half (t : ℝ) : energyDeviation (1 / 2) t = 0 := by
  unfold energyDeviation
  simp only [ofReal_div, ofReal_one, ofReal_ofNat]
  ring

/-- ΔE(σ,t) = ΔE(1-σ,-t) from the functional equation ξ(1-s)=ξ(s). -/
theorem energyDeviation_symm_neg (σ t : ℝ) :
    energyDeviation σ t = energyDeviation (1 - σ) (-t) := by
  unfold energyDeviation
  have h1 : xi (↑σ + ↑t * I) = xi (↑(1 - σ) + ↑(-t) * I) := by
    have : (↑σ : ℂ) + ↑t * I = 1 - (↑(1 - σ) + ↑(-t) * I) := by push_cast; ring
    conv_lhs => rw [this, xi_functional_eq]
  have h2 : xi (1 / 2 + ↑t * I) = xi (1 / 2 + ↑(-t) * I) := by
    have : (1 / 2 : ℂ) + ↑t * I = 1 - (1 / 2 + ↑(-t) * I) := by push_cast; ring
    conv_lhs => rw [this, xi_functional_eq]
  rw [show (↑(1 - σ) : ℂ) + ↑(-t) * I = ↑(1 - σ) + ↑(-t) * I from rfl]
  simp only [h1, h2]

/-- ΔE(σ,t) = ΔE(1-σ,t) assuming conjugation symmetry ξ(conj s) = conj(ξ(s)). -/
theorem energyDeviation_symm
    (hconj : ∀ s : ℂ, xi (starRingEnd ℂ s) = starRingEnd ℂ (xi s))
    (σ t : ℝ) :
    energyDeviation σ t = energyDeviation (1 - σ) t := by
  rw [energyDeviation_symm_neg]
  unfold energyDeviation
  congr 1
  · congr 1
    have h1 : (↑(1 - σ) : ℂ) + ↑(-t) * I = starRingEnd ℂ (↑(1 - σ) + ↑t * I) := by
      have : starRingEnd ℂ (↑(1 - σ) + ↑t * I) = ↑(1 - σ) - ↑t * I := by
        rw [map_add, conj_ofReal, map_mul, conj_ofReal, conj_I]; ring
      rw [this]; push_cast; ring
    rw [h1, hconj, RCLike.norm_conj]
  · congr 1
    have h1 : (1 / 2 : ℂ) + ↑(-t) * I = starRingEnd ℂ (1 / 2 + ↑t * I) := by
      have : starRingEnd ℂ ((1:ℂ)/2 + ↑t * I) = 1/2 - ↑t * I := by
        rw [map_add, map_div₀, map_one, map_ofNat, map_mul, conj_ofReal, conj_I]; ring
      rw [this]; push_cast; ring
    rw [h1, hconj, RCLike.norm_conj]

/-
============================================================
Section 4: Variational Identity
============================================================

The variational energy identity: ∂_σ log‖f(σ+it₀)‖ = Re[f'/f(σ₀+it₀)]
    when f is differentiable and nonzero.
-/
theorem variational_energy_identity
    (f : ℂ → ℂ) (σ₀ t₀ : ℝ) (hf : DifferentiableAt ℂ f (↑σ₀ + ↑t₀ * I))
    (hnz : f (↑σ₀ + ↑t₀ * I) ≠ 0) :
    HasDerivAt (fun σ : ℝ => Real.log ‖f (↑σ + ↑t₀ * I)‖)
      ((logDeriv f (↑σ₀ + ↑t₀ * I)).re) σ₀ := by
  have h_deriv : HasDerivAt (fun σ : ℝ => f (σ + t₀ * Complex.I)) (deriv f (σ₀ + t₀ * Complex.I)) σ₀ := by
    convert HasDerivAt.comp σ₀ ( hf.hasDerivAt ) ( HasDerivAt.add ( hasDerivAt_id _ |> HasDerivAt.ofReal_comp ) ( hasDerivAt_const _ _ ) ) using 1 ; aesop;
  have := h_deriv.norm_sq;
  convert HasDerivAt.log ( this.sqrt ?_ ) ?_ using 1 <;> norm_num [ hnz ];
  unfold logDeriv; norm_num [ Complex.normSq, Complex.norm_def ] ; ring;
  norm_num [ Complex.normSq, Complex.inv_re, Complex.inv_im, Real.sq_sqrt <| add_nonneg ( sq_nonneg _ ) ( sq_nonneg _ ) ] ; ring

/-
============================================================
Section 5: Logarithmic Derivative Antisymmetry
============================================================

For any f with f(1-s) = f(s) and differentiable at (1-s),
    logDeriv f (1-s) = -logDeriv f s.
-/
theorem logDeriv_neg_of_functional_eq
    {f : ℂ → ℂ} (hfeq : ∀ s, f (1 - s) = f s)
    {s : ℂ} (hf : DifferentiableAt ℂ f (1 - s)) :
    logDeriv f (1 - s) = -logDeriv f s := by
  have h_chain : deriv (fun x => f (1 - x)) s = deriv f (1 - s) * deriv (fun x => 1 - x) s := by
    exact deriv_comp s hf ( differentiableAt_id.const_sub _ );
  by_cases H : f s = 0 <;> simp_all +decide [ sub_eq_add_neg, logDeriv ];
  ring

/-- Re[ξ'/ξ(σ+it)] = -Re[ξ'/ξ((1-σ)+i(-t))] from the functional equation alone. -/
theorem logDerivRe_odd_symmetry_neg
    {σ t : ℝ}
    (hd : DifferentiableAt ℂ xi (↑(1 - σ) + ↑(-t) * I)) :
    (logDeriv xi (↑σ + ↑t * I)).re =
    -(logDeriv xi ((↑(1 - σ) : ℂ) + ↑(-t) * I)).re := by
  have heq : (1 : ℂ) - (↑σ + ↑t * I) = ↑(1 - σ) + ↑(-t) * I := by push_cast; ring
  have hd' : DifferentiableAt ℂ xi (1 - (↑σ + ↑t * I)) := by rwa [heq]
  have key : logDeriv xi (1 - (↑σ + ↑t * I)) = -logDeriv xi (↑σ + ↑t * I) :=
    logDeriv_neg_of_functional_eq xi_functional_eq hd'
  rw [heq] at key
  linarith [congrArg Complex.re key, neg_re (logDeriv xi (↑σ + ↑t * I))]

/-
Re[ξ'/ξ(σ+it)] = -Re[ξ'/ξ((1-σ)+it)] (full odd symmetry).
    Requires both the functional equation and conjugation symmetry.
-/
theorem logDerivRe_odd_symmetry
    (hconj : ∀ s : ℂ, xi (starRingEnd ℂ s) = starRingEnd ℂ (xi s))
    {σ t : ℝ}
    (hd : DifferentiableAt ℂ xi (↑(1 - σ) + ↑(-t) * I))
    (hd' : DifferentiableAt ℂ xi (↑(1 - σ) + ↑t * I)) :
    (logDeriv xi (↑σ + ↑t * I)).re =
    -(logDeriv xi ((↑(1 - σ) : ℂ) + ↑t * I)).re := by
  -- Use logDerivRe_odd_symmetry_neg with hd:
  have h1 := logDerivRe_odd_symmetry_neg hd;
  -- From hconj, for any differentiable function satisfying the conjugation relation, we have logDeriv f (conj z) = conj(logDeriv f z), hence Re[logDeriv f (conj z)] = Re[logDeriv f z].
  have h_logDeriv_conj : ∀ z : ℂ, DifferentiableAt ℂ xi z → logDeriv xi (starRingEnd ℂ z) = starRingEnd ℂ (logDeriv xi z) := by
    intros z hz
    have h_deriv_conj : deriv xi (starRingEnd ℂ z) = starRingEnd ℂ (deriv xi z) := by
      refine' HasDerivAt.deriv _;
      rw [ hasDerivAt_iff_tendsto_slope_zero ];
      have h_deriv_conj : Filter.Tendsto (fun t => (xi (z + t) - xi z) / t) (nhdsWithin 0 {0}ᶜ) (nhds (deriv xi z)) := by
        simpa [ div_eq_inv_mul ] using hz.hasDerivAt.tendsto_slope_zero;
      convert Complex.continuous_conj.continuousAt.tendsto.comp ( h_deriv_conj.comp ( show Filter.Tendsto ( fun t : ℂ => starRingEnd ℂ t ) ( nhdsWithin 0 { 0 } ᶜ ) ( nhdsWithin 0 { 0 } ᶜ ) from ?_ ) ) using 2 <;> norm_num [ div_eq_inv_mul ];
      · exact Or.inl ( by rw [ ← hconj, ← hconj ] ; simp +decide [ add_comm ] );
      · rw [ Metric.tendsto_nhdsWithin_nhdsWithin ] ; aesop;
    unfold logDeriv; aesop;
  specialize h_logDeriv_conj ( ( 1 - σ ) + t * Complex.I ) ; aesop;

/-- On the critical line, Re[ξ'/ξ(1/2+it)] = 0. -/
theorem logDerivRe_vanishes_on_critical_line
    (hconj : ∀ s : ℂ, xi (starRingEnd ℂ s) = starRingEnd ℂ (xi s))
    {t : ℝ}
    (hd : DifferentiableAt ℂ xi (1 / 2 + ↑(-t) * I))
    (hd' : DifferentiableAt ℂ xi (1 / 2 + ↑t * I)) :
    (logDeriv xi (1 / 2 + ↑t * I)).re = 0 := by
  have h1c : (↑(1 - 1 / 2 : ℝ) : ℂ) = 1 / 2 := by push_cast; ring
  have hd2 : DifferentiableAt ℂ xi (↑(1 - 1 / 2 : ℝ) + ↑(-t) * I) := by
    rwa [show (↑(1 - 1 / 2 : ℝ) : ℂ) + ↑(-t) * I = 1 / 2 + ↑(-t) * I from by rw [h1c]]
  have hd2' : DifferentiableAt ℂ xi (↑(1 - 1 / 2 : ℝ) + ↑t * I) := by
    rwa [show (↑(1 - 1 / 2 : ℝ) : ℂ) + ↑t * I = 1 / 2 + ↑t * I from by rw [h1c]]
  have h := logDerivRe_odd_symmetry hconj (σ := 1 / 2) (t := t) hd2 hd2'
  rw [show (↑(1 / 2 : ℝ) : ℂ) = (1 : ℂ) / 2 from by push_cast; ring] at h
  rw [h1c] at h
  linarith

/-
============================================================
Section 6: Equivalence of Gap Strip Problems
============================================================

For an odd function g (g(1-σ) = -g(σ)):
    g > 0 on (1/2, b] ↔ g < 0 on [1-b, 1/2).
-/
theorem equivalence_pos_neg_strip
    {g : ℝ → ℝ} (hg : ∀ σ, g (1 - σ) = -g σ)
    {b : ℝ} (_hb : b > 1 / 2) :
    (∀ σ, 1 / 2 < σ → σ ≤ b → g σ > 0) ↔
    (∀ σ, 1 - b ≤ σ → σ < 1 / 2 → g σ < 0) := by
  grind +locals

/-- Closing the gap on the right (g > 0 for σ ∈ (1/2, gapHi])
    is equivalent to g < 0 for σ ∈ [gapLo, 1/2). -/
theorem equivalence_gap_strip
    {g : ℝ → ℝ} (hg : ∀ σ, g (1 - σ) = -g σ) :
    (∀ σ, 1 / 2 < σ → σ ≤ gapHi → g σ > 0) ↔
    (∀ σ, gapLo ≤ σ → σ < 1 / 2 → g σ < 0) := by
  rw [gapLo_eq_one_sub_gapHi]
  exact equivalence_pos_neg_strip hg gapHi_gt_half

/-
============================================================
Section 7: Norm and Energy Deviation Relationships
============================================================

If ΔE(σ,t) > 0 and both norms are positive,
    then ‖ξ(σ+it)‖ > ‖ξ(1/2+it)‖.
-/
theorem norm_gt_of_energyDeviation_pos {σ t : ℝ}
    (hΔ : energyDeviation σ t > 0)
    (hpos_s : ‖xi (↑σ + ↑t * I)‖ > 0)
    (_hpos : ‖xi (1 / 2 + ↑t * I)‖ > 0) :
    ‖xi (↑σ + ↑t * I)‖ > ‖xi (1 / 2 + ↑t * I)‖ := by
  contrapose! hΔ;
  exact sub_nonpos_of_le ( Real.log_le_log hpos_s hΔ )

/-
ΔE(σ,t) ≥ 0 ↔ ‖ξ(σ+it)‖ ≥ ‖ξ(1/2+it)‖, when both norms are positive.
-/
theorem energyDeviation_nonneg_iff {σ t : ℝ}
    (hpos_s : ‖xi (↑σ + ↑t * I)‖ > 0)
    (hpos : ‖xi (1 / 2 + ↑t * I)‖ > 0) :
    energyDeviation σ t ≥ 0 ↔
    ‖xi (↑σ + ↑t * I)‖ ≥ ‖xi (1 / 2 + ↑t * I)‖ := by
  rw [ energyDeviation ];
  rw [ ge_iff_le, sub_nonneg, Real.log_le_log_iff ] <;> aesop

/-- If ξ(σ+it) = 0 and ‖ξ(1/2+it)‖ ≥ 1, then ΔE(σ,t) ≤ 0.
    Uses Real.log 0 = 0 convention. -/
theorem energyDeviation_nonpos_of_zero {σ t : ℝ}
    (hzero : xi (↑σ + ↑t * I) = 0)
    (hbase : ‖xi (1 / 2 + ↑t * I)‖ ≥ 1) :
    energyDeviation σ t ≤ 0 := by
  unfold energyDeviation
  rw [norm_eq_zero.mpr hzero]; norm_num; linarith [Real.log_nonneg hbase]

-- ============================================================
-- Section 8: Outer Boundary (conditional on Guth–Maynard)
-- ============================================================

/-- ΔE(σ,t) > 0 for σ > 1/2, conditional on the hypothesis that
    the function σ ↦ log‖ξ(σ+it)‖ is strictly monotone increasing.
    This monotonicity follows from Re[ξ'/ξ] > 0 (Hadamard product + Guth–Maynard). -/
theorem energyDeviation_pos_of_increasing
    {σ t : ℝ} (hσ : σ > 1 / 2)
    (hmono : StrictMono (fun u : ℝ => Real.log ‖xi (↑u + ↑t * I)‖)) :
    energyDeviation σ t > 0 := by
  unfold energyDeviation
  have key : Real.log ‖xi (↑(1/2 : ℝ) + ↑t * I)‖ < Real.log ‖xi (↑σ + ↑t * I)‖ :=
    hmono hσ
  have h : (↑(1/2 : ℝ) : ℂ) + ↑t * I = (1:ℂ)/2 + ↑t * I := by push_cast; ring
  rw [h] at key
  linarith

/-
============================================================
Section 9: Symmetric Pair Reinforcement
============================================================

Re(z⁻¹) > 0 when Re(z) > 0 and z ≠ 0.
-/
theorem re_inv_pos_of_re_pos {z : ℂ} (hz : z ≠ 0) (hre : z.re > 0) :
    z⁻¹.re > 0 := by
  simp_all +decide [ Complex.inv_re ]

/-- Symmetric pair reinforcement: for z₁, z₂ with positive real parts,
    Re(z₁⁻¹) + Re(z₂⁻¹) > 0. -/
theorem symmetric_pair_reinforcement
    {z₁ z₂ : ℂ} (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0)
    (h₁ : z₁.re > 0) (h₂ : z₂.re > 0) :
    z₁⁻¹.re + z₂⁻¹.re > 0 :=
  add_pos (re_inv_pos_of_re_pos hz₁ h₁) (re_inv_pos_of_re_pos hz₂ h₂)

/-- Geometric distance positivity for symmetric pair. -/
theorem paired_zero_distances_pos {σ_test σ₀ γ₀ : ℝ}
    (htest_ne_σ₀ : σ_test ≠ σ₀)
    (htest_ne_mirror : σ_test ≠ 1 - σ₀) :
    (σ_test - σ₀) ^ 2 + (0 - γ₀) ^ 2 > 0 ∧
    (σ_test - (1 - σ₀)) ^ 2 + (0 - γ₀) ^ 2 > 0 := by
  exact ⟨by nlinarith [mul_self_pos.2 (sub_ne_zero.2 htest_ne_σ₀)],
         by nlinarith [mul_self_pos.2 (sub_ne_zero.2 htest_ne_mirror)]⟩

-- ============================================================
-- Section 10: Zero Exclusion
-- ============================================================

/-- If ΔE > 0 throughout (1/2, gapHi] and ‖ξ(1/2+it)‖ ≥ 1 for all t,
    then ξ has no zeros with Re(s) ∈ (1/2, gapHi]. -/
theorem no_zeros_in_gap_of_energyDeviation_pos
    (hΔ : ∀ σ t : ℝ, 1 / 2 < σ → σ ≤ gapHi → energyDeviation σ t > 0)
    (hbase : ∀ t : ℝ, ‖xi (1 / 2 + ↑t * I)‖ ≥ 1)
    {σ t : ℝ} (hσ1 : 1 / 2 < σ) (hσ2 : σ ≤ gapHi) :
    xi (↑σ + ↑t * I) ≠ 0 := by
  by_contra h_zero
  exact absurd (hΔ σ t hσ1 hσ2)
    (not_lt_of_ge (energyDeviation_nonpos_of_zero h_zero (hbase t)))

/-
Extension to (gapLo, gapHi) \ {1/2} using the functional equation.
-/
theorem kappa_vanishes_of_energyDeviation_pos
    (hΔ : ∀ σ t : ℝ, 1 / 2 < σ → σ ≤ gapHi → energyDeviation σ t > 0)
    (hbase : ∀ t : ℝ, ‖xi (1 / 2 + ↑t * I)‖ ≥ 1)
    {σ t : ℝ} (hσ1 : gapLo < σ) (hσ2 : σ < gapHi)
    (hne : σ ≠ 1 / 2) :
    xi (↑σ + ↑t * I) ≠ 0 := by
  have h_case2 : xi (↑σ + ↑t * I) = xi (↑(1 - σ) + ↑(-t) * I) := by
    convert xi_functional_eq _ using 2 ; push_cast ; ring;
  by_cases hcase : σ > 1 / 2;
  · convert no_zeros_in_gap_of_energyDeviation_pos hΔ hbase hcase hσ2.le using 1;
  · convert no_zeros_in_gap_of_energyDeviation_pos hΔ hbase ( show 1 / 2 < 1 - σ by contrapose! hne; linarith ) ( show 1 - σ ≤ gapHi by linarith [ show gapLo = 1 - gapHi by norm_num [ gapLo, gapHi ] ] ) using 1

/-
============================================================
Section 11: Open Questions
============================================================

OQ-M-75: The energy deviation is nonneg in the gap strip.
    This requires the Hadamard product representation (not in Mathlib),
    so we take as hypothesis that ‖ξ‖ is weakly increasing in σ for σ > 1/2.
-/
theorem OQ_M_75_from_monotone
    (hmono : ∀ t : ℝ, ∀ σ₁ σ₂ : ℝ, 1 / 2 ≤ σ₁ → σ₁ ≤ σ₂ →
      Real.log ‖xi (↑σ₁ + ↑t * I)‖ ≤ Real.log ‖xi (↑σ₂ + ↑t * I)‖) :
    ∀ σ t : ℝ, 1 / 2 < σ → σ ≤ gapHi → energyDeviation σ t ≥ 0 := by
  -- Apply the monotonicity hypothesis hmono with σ₁ = 1/2 and σ₂ = σ.
  intros σ t hσ1 hσ2
  have h_log : Real.log ‖xi ((1 / 2 : ℝ) + t * I)‖ ≤ Real.log ‖xi (σ + t * I)‖ := by
    exact hmono t _ _ le_rfl hσ1.le;
  exact sub_nonneg_of_le <| by simpa using h_log;

/-- OQ-L-1: Combining asymptotic and finite checks for ‖ξ(1/2+it)‖ ≥ 1. -/
theorem OQ_L_1_asymptotic_bound
    {T₀ : ℝ}
    (hgrowth : ∀ t : ℝ, |t| ≥ T₀ → ‖xi (1 / 2 + ↑t * I)‖ ≥ 1)
    (hfinite : ∀ t : ℝ, |t| < T₀ → ‖xi (1 / 2 + ↑t * I)‖ ≥ 1) :
    ∀ t : ℝ, ‖xi (1 / 2 + ↑t * I)‖ ≥ 1 := by
  exact fun t => if ht : |t| < T₀ then hfinite t ht else hgrowth t (le_of_not_gt ht)

/-- CJ-14: Energy reinforcement / no zeros in gap. -/
def buchanan_energy_reinforcement := @no_zeros_in_gap_of_energyDeviation_pos

end MNZI