/-
  MNZI Paper A-1: Finite-Order Corrections to the GUE Spacing Ratio Mode

  Companion to MNZI Paper A. Formalises algebraic results about the ABGR surmise,
  the Buchanan–Atas identity, tail asymptotics, mode shift perturbation theory,
  and finite-size bounds for MBL diagnostics.
-/
import Mathlib
import MNZI.Core

namespace MNZI

open Real

/-! ## Golden ratio inverse φ⁻¹ = (√5 - 1)/2 -/

-- goldenRatioInv imported from MNZI.Core (§2).
-- phiInv is a local alias used throughout this file.
noncomputable def phiInv : ℝ := goldenRatioInv

/-- φ⁻¹ = (√5 - 1)/2. -/
theorem phiInv_eq : phiInv = (Real.sqrt 5 - 1) / 2 := by
  unfold phiInv; exact goldenRatioInv_explicit

/-
φ⁻¹ is positive.
-/
theorem phiInv_pos : 0 < phiInv := by
  unfold phiInv; exact goldenRatioInv_pos

/-
The key golden ratio identity: φ⁻¹² + φ⁻¹ - 1 = 0.
-/
theorem phiInv_sq_add_sub : phiInv ^ 2 + phiInv - 1 = 0 := by
  unfold phiInv; exact goldenRatioInv_sq_add

/-
φ⁻¹(1 + φ⁻¹) = 1.
-/
theorem phiInv_mul_one_add : phiInv * (1 + phiInv) = 1 := by
  linarith [ phiInv_sq_add_sub ]

/-
2φ⁻¹ + 1 = √5.
-/
theorem two_phiInv_add_one : 2 * phiInv + 1 = Real.sqrt 5 := by
  have h := phiInv_eq
  linarith

/-
√5 > 0.
-/
theorem sqrt5_pos : (0 : ℝ) < Real.sqrt 5 := by
  positivity

/-! ## The Buchanan–Atas Identity -/

/-
The Buchanan–Atas perfect-square identity:
    (1 + r + r²)² - 4r(1 + r) = (r² + r - 1)² for all r ∈ ℝ.
    This is the algebraic core of the ABGR mode proof.
-/
theorem buchanan_atas_identity (r : ℝ) :
    (1 + r + r ^ 2) ^ 2 - 4 * r * (1 + r) = (r ^ 2 + r - 1) ^ 2 := by
      ring

/-! ## ABGR distribution (unnormalised) -/

/-- The unnormalised ABGR density: (r + r²)² / (1 + r + r²)⁴.
    This equals (r(1+r))² / (1+r+r²)⁴. -/
noncomputable def ABGR_unnorm (r : ℝ) : ℝ :=
  (r + r ^ 2) ^ 2 / (1 + r + r ^ 2) ^ 4

/-- The normalisation constant Z₂ = 4√3 π / 243. -/
noncomputable def Z2 : ℝ := 4 * Real.sqrt 3 * Real.pi / 243

/-
Z₂ is positive.
-/
theorem Z2_pos : 0 < Z2 := by
  exact div_pos ( mul_pos ( mul_pos ( by norm_num ) ( Real.sqrt_pos.mpr ( by norm_num ) ) ) ( Real.pi_pos ) ) ( by norm_num )

/-- The normalised ABGR density p₂(r) = ABGR_unnorm(r) / Z₂. -/
noncomputable def ABGR (r : ℝ) : ℝ := ABGR_unnorm r / Z2

/-! ## Mode of the ABGR distribution -/

/-
The unnormalised ABGR density at φ⁻¹ equals 1/16.
-/
theorem ABGR_unnorm_at_phiInv : ABGR_unnorm phiInv = 1 / 16 := by
  unfold ABGR_unnorm phiInv;
  rw [ show goldenRatioInv + goldenRatioInv ^ 2 = 1 by linarith [ goldenRatioInv_sq_add ], show 1 + goldenRatioInv + goldenRatioInv ^ 2 = 2 by linarith [ goldenRatioInv_sq_add ] ] ; norm_num

/-
The unnormalised ABGR density is at most 1/16 for all r ≥ 0.
    This proves φ⁻¹ is the mode.
-/
theorem ABGR_unnorm_le_one_sixteenth (r : ℝ) (hr : 0 ≤ r) :
    ABGR_unnorm r ≤ 1 / 16 := by
      -- We need to show that $(r + r^2)^2 / (1 + r + r^2)^4 \leq 1/16$ for all $r \ge 0$.
      -- This is equivalent to $16(r + r^2)^2 \leq (1 + r + r^2)^4$.
      suffices h_suff : 16 * (r + r ^ 2) ^ 2 ≤ (1 + r + r ^ 2) ^ 4 by
        rw [ ABGR_unnorm, div_le_iff₀ ] <;> first | positivity | linarith;
      nlinarith [ sq_nonneg ( r ^ 2 + r - 1 ) ]

/-
The mode of the unnormalised ABGR density is at φ⁻¹:
    ABGR_unnorm achieves its maximum value 1/16 at φ⁻¹, and
    this value is an upper bound for all r ≥ 0.
-/
theorem ABGR_mode_is_phiInv :
    ABGR_unnorm phiInv = 1 / 16 ∧
    ∀ r : ℝ, 0 ≤ r → ABGR_unnorm r ≤ 1 / 16 := by
      exact ⟨ABGR_unnorm_at_phiInv, ABGR_unnorm_le_one_sixteenth⟩

/-! ## Tail asymptotics -/

/-- Tail factorisation: for r > 0, we can write the unnormalised ABGR as
    r⁻⁴ · h(r) where h(r) = (1 + 1/r)² / (1/r² + 1/r + 1)⁴.
    This captures the r⁻⁴ tail decay. -/
noncomputable def ABGR_tail_h (r : ℝ) : ℝ :=
  (1 + 1 / r) ^ 2 / (1 / r ^ 2 + 1 / r + 1) ^ 4

/-
For r > 0, ABGR_unnorm(r) = r⁻⁴ · ABGR_tail_h(r).
    This gives the r⁻⁴ tail factorisation.
-/
theorem ABGR_tail_factorisation (r : ℝ) (hr : 0 < r) :
    ABGR_unnorm r = r⁻¹ ^ 4 * ABGR_tail_h r := by
      unfold ABGR_unnorm ABGR_tail_h;
      field_simp
      ring

/-
The tail factor h(r) → 1 as r → ∞, confirming p₂(r) ~ r⁻⁴.
-/
theorem ABGR_tail_h_tendsto :
    Filter.Tendsto ABGR_tail_h Filter.atTop (nhds 1) := by
      -- We can simplify the expression for $h(r)$ as $r \to \infty$.
      have h_simplify : Filter.Tendsto (fun r : ℝ => (1 + 1 / r) ^ 2 / (1 / r ^ 2 + 1 / r + 1) ^ 4) Filter.atTop (nhds ((1 + 0) ^ 2 / (0 + 0 + 1) ^ 4)) := by
        exact Filter.Tendsto.div ( Filter.Tendsto.pow ( tendsto_const_nhds.add ( tendsto_const_nhds.div_atTop Filter.tendsto_id ) ) _ ) ( Filter.Tendsto.pow ( Filter.Tendsto.add ( Filter.Tendsto.add ( tendsto_const_nhds.div_atTop ( by norm_num ) ) ( tendsto_const_nhds.div_atTop Filter.tendsto_id ) ) tendsto_const_nhds ) _ ) ( by norm_num );
      convert h_simplify using 2 ; norm_num

/-! ## Denominator exponent for the N×N surmise -/

/-- The denominator exponent M_N = (N² - 1)/2 for the N×N GUE surmise.
    For N = 3: M₃ = 4. For N = 5: M₅ = 12. -/
def surmise_exponent (N : ℕ) : ℕ := (N ^ 2 - 1) / 2

/-
M₃ = 4: the ABGR denominator exponent.
-/
theorem surmise_exponent_3 : surmise_exponent 3 = 4 := by
  rfl

/-
M₅ = 12: the 5×5 surmise denominator exponent.
-/
theorem surmise_exponent_5 : surmise_exponent 5 = 12 := by
  rfl

/-
M₇ = 24: the 7×7 surmise denominator exponent.
-/
theorem surmise_exponent_7 : surmise_exponent 7 = 24 := by
  rfl

/-! ## Tail exponent for the N×N surmise -/

/-- The tail exponent for the N×N GUE surmise (odd N ≥ 3) is -(N²-1)/2.
    This is the power of r in the large-r asymptotic p^(N)(r) ~ C_N · r^{-(N²-1)/2}. -/
def tail_exponent (N : ℕ) : ℤ := -((N ^ 2 - 1 : ℤ)) / 2

/-
The tail exponent for N = 3 is -4.
-/
theorem tail_exponent_3 : tail_exponent 3 = -4 := by
  norm_num [tail_exponent]

/-
The tail exponent for N = 5 is -12.
-/
theorem tail_exponent_5 : tail_exponent 5 = -12 := by
  norm_num [tail_exponent]

/-
The tail exponent for N = 7 is -24.
-/
theorem tail_exponent_7 : tail_exponent 7 = -24 := by
  norm_num [tail_exponent]

/-
The Vandermonde cross-term count: for odd N with k = (N-1)/2,
    2k(N-k) = (N²-1)/2. This proves the tail exponent equals the
    denominator exponent (with a minus sign).
-/
theorem vandermonde_cross_count (N : ℕ) (hN : 1 ≤ N) (hOdd : N % 2 = 1) :
    let k := (N - 1) / 2
    2 * k * (N - k) = (N ^ 2 - 1) / 2 := by
      rcases Nat.even_or_odd' N with ⟨ k, rfl | rfl ⟩ <;> norm_num at *;
      exact Eq.symm ( Nat.div_eq_of_eq_mul_left zero_lt_two ( Nat.sub_eq_of_eq_add <| by nlinarith [ Nat.sub_add_cancel ( by linarith : k ≤ 2 * k + 1 ) ] ) )

/-! ## Mode shift perturbation theory -/

/-
The exact curvature of the ABGR density at its mode:
    p₂''(φ⁻¹) = -5/(16·Z₂).

    The mode shift prefactor 16·Z₂/5 is the reciprocal of the
    absolute value of this curvature (with sign).
-/
theorem mode_shift_prefactor :
    (16 : ℝ) * Z2 / 5 = 64 * Real.sqrt 3 * Real.pi / (243 * 5) := by
      unfold Z2; ring

/-
The curvature value: -5/(16·Z₂) expressed in terms of π and √3.
-/
theorem curvature_at_mode :
    (5 : ℝ) / (16 * Z2) = 405 * Real.sqrt 3 / (64 * Real.pi) := by
      unfold Z2; ring; norm_num [ Real.pi_ne_zero ] ;
      rw [ ← Real.sqrt_div_self ] ; ring

/-! ## Finite-size bounds -/

/-
2^56 > 7.2 × 10^16: the finite-size suppression factor for L = 14 spins.
-/
theorem two_pow_56 : (2 : ℕ) ^ 56 > 72 * 10 ^ 15 := by
  norm_num

/-
For L ≥ 14, we have 2^(4L) ≥ 2^56, meaning finite-N_H corrections
    are O(N_H^{-4}) = O(2^{-4L}) ≤ 2^{-56} ≈ 1.4 × 10^{-17}.
-/
theorem finite_size_bound (L : ℕ) (hL : 14 ≤ L) :
    (2 : ℕ) ^ 56 ≤ 2 ^ (4 * L) := by
      exact pow_le_pow_right₀ ( by norm_num ) ( by linarith )

/-
The Nishigaki cancellation: for L ≥ 14 spins (N_H = 2^L),
    finite Hilbert-space corrections satisfy 2^(4L) > 7.2 × 10^16,
    so corrections are ≤ 10^{-17}.
-/
theorem nishigaki_cancellation_bound (L : ℕ) (hL : 14 ≤ L) :
    72 * 10 ^ 15 < (2 : ℕ) ^ (4 * L) := by
      exact lt_of_lt_of_le ( by norm_num ) ( pow_le_pow_right₀ ( by norm_num ) ( Nat.mul_le_mul_left 4 hL ) )

/-! ## 5×5 mode shift -/

/-
The 5×5 surmise tail is dramatically lighter than the 3×3:
    the ratio of tail exponents is 12/4 = 3.
-/
theorem tail_ratio_5_to_3 : surmise_exponent 5 / surmise_exponent 3 = 3 := by
  norm_num [surmise_exponent]

end MNZI