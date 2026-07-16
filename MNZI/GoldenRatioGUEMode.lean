/-
  MNZI Paper A: The Golden Ratio as the Mode of the GUE Consecutive
  Spacing Ratio Distribution

  This file proves that the mode of the GUE consecutive level-spacing
  ratio distribution (Atas, Bogomolny, Giraud, Roux 2013) is exactly
  φ⁻¹ = (√5 - 1)/2, the reciprocal of the golden ratio.

  The proof is purely algebraic: the Buchanan–Atas identity
    (1 + r + r²)² - 4r(1 + r) = (r² + r - 1)²
  shows that the Atas density satisfies p₂(r) ≤ 1/16 for all r ≥ 0,
  with equality if and only if r² + r - 1 = 0, whose unique positive
  root is φ⁻¹. No differentiation is required.
-/

import Mathlib
import MNZI.Core

namespace MNZI

open Real

-- goldenRatioInv, goldenRatioInv_pos, goldenRatioInv_lt_one,
-- goldenRatioInv_explicit, goldenRatioInv_sq_add
-- imported from MNZI.Core (§2).

/-- The golden ratio: φ = (1 + √5)/2. -/
noncomputable def goldenRatio : ℝ := (1 + Real.sqrt 5) / 2

/-! ## The Buchanan–Atas algebraic identity -/

/-
The Buchanan–Atas identity: (1 + r + r²)² - 4r(1 + r) = (r² + r - 1)²
    for all r : ℝ. This is verified by direct expansion.
-/
theorem algebraic_identity (r : ℝ) :
    (1 + r + r ^ 2) ^ 2 - 4 * r * (1 + r) = (r ^ 2 + r - 1) ^ 2 := by
  ring

/-! ## Properties of the golden ratio inverse -/

/-
√5 > 0
-/
lemma sqrt5_pos : Real.sqrt 5 > 0 := by
  positivity

/-
√5 > 1
-/
lemma sqrt5_gt_one : Real.sqrt 5 > 1 := by
  exact Real.lt_sqrt_of_sq_lt ( by norm_num )

/-
(√5)² = 5
-/
lemma sq_sqrt5 : Real.sqrt 5 ^ 2 = 5 := by
  norm_num

-- goldenRatioInv_root (= Core's goldenRatioInv_sq_add) imported from Core.
-- goldenRatioInv_pos imported from Core.

/-
If x > 0 and x² + x - 1 = 0, then x = φ⁻¹.
-/
theorem goldenRatioInv_unique_pos_root (x : ℝ) (hx : x > 0) (hroot : x ^ 2 + x - 1 = 0) :
    x = goldenRatioInv := by
  rw [goldenRatioInv_explicit]
  nlinarith [Real.sqrt_nonneg 5, Real.sq_sqrt (show 0 ≤ 5 by norm_num)]

/-! ## The Atas density function (unnormalized) -/

/-- The (unnormalized) Atas GUE density:
    p₂(r) = (r(1+r) / (1+r+r²)²)² for r ≥ 0.
    This captures the shape; the normalization constant Z₂ is
    a positive scalar that does not affect the mode location. -/
noncomputable def atasDensity (r : ℝ) : ℝ :=
  (r * (1 + r) / (1 + r + r ^ 2) ^ 2) ^ 2

/-! ## Upper bound from the algebraic identity -/

/-
For all r ≥ 0, 4r(1+r) ≤ (1+r+r²)².
-/
theorem atas_sqrt_density_le (r : ℝ) (_hr : r ≥ 0) :
    4 * r * (1 + r) ≤ (1 + r + r ^ 2) ^ 2 := by
  nlinarith [ sq_nonneg ( r^2 + r - 1 ) ]

/-
For r ≥ 0, equality 4r(1+r) = (1+r+r²)² holds iff r²+r-1 = 0.
-/
theorem atas_sqrt_density_eq_iff (r : ℝ) (_hr : r ≥ 0) :
    4 * r * (1 + r) = (1 + r + r ^ 2) ^ 2 ↔ r ^ 2 + r - 1 = 0 := by
  grind +qlia

/-! ## Density value and bounds -/

/-
1 + goldenRatioInv + goldenRatioInv² ≠ 0
-/
lemma one_add_phi_add_phi_sq_ne_zero :
    1 + goldenRatioInv + goldenRatioInv ^ 2 ≠ 0 := by
  exact ne_of_gt ( add_pos ( add_pos zero_lt_one ( goldenRatioInv_pos ) ) ( sq_pos_of_pos ( goldenRatioInv_pos ) ) )

/-
The Atas density at φ⁻¹ equals 1/16.
-/
theorem atasDensity_at_goldenRatioInv :
    atasDensity goldenRatioInv = 1 / 16 := by
  convert congr_arg ( · ^ 2 ) ( show goldenRatioInv * ( 1 + goldenRatioInv ) / ( 1 + goldenRatioInv + goldenRatioInv ^ 2 ) ^ 2 = 1 / 4 by
                                  rw [ div_eq_iff ] <;> nlinarith [ goldenRatioInv_pos, goldenRatioInv_sq_add ] ) using 1 ; norm_num [ div_pow ]

/-
Helper: 1 + r + r² > 0 for r ≥ 0.
-/
lemma one_add_r_add_r_sq_pos (r : ℝ) (hr : r ≥ 0) :
    1 + r + r ^ 2 > 0 := by
  positivity

/-
The Atas density satisfies p₂(r) ≤ 1/16 for all r ≥ 0.
-/
theorem atasDensity_le (r : ℝ) (hr : r ≥ 0) :
    atasDensity r ≤ 1 / 16 := by
  norm_num [ atasDensity ];
  rw [ div_pow, div_le_div_iff₀ ] <;> try positivity;
  nlinarith [ sq_nonneg ( r ^ 2 + r - 1 ) ]

/-
For r > 0, p₂(r) = 1/16 iff r = φ⁻¹.
-/
theorem atasDensity_eq_max_iff (r : ℝ) (hr : r > 0) :
    atasDensity r = 1 / 16 ↔ r = goldenRatioInv := by
  constructor;
  · intro h_eq
    have h_eq' : r * (1 + r) / (1 + r + r ^ 2) ^ 2 = 1 / 4 := by
      unfold atasDensity at h_eq; rw [ ← sq_eq_sq₀ ] <;> first | positivity | linarith;
    exact goldenRatioInv_unique_pos_root r hr ( by rw [ div_eq_iff ( by positivity ) ] at h_eq'; nlinarith );
  · exact fun h => h.symm ▸ atasDensity_at_goldenRatioInv

/-! ## The main theorem -/

/-
**Main Theorem**: The mode of the GUE consecutive spacing ratio
    distribution is uniquely attained at φ⁻¹ = (√5-1)/2.
    That is, for all r > 0 with r ≠ φ⁻¹, p₂(r) < p₂(φ⁻¹) = 1/16.
-/
theorem gue_mode_is_golden_ratio_inv (r : ℝ) (hr : r > 0) (hne : r ≠ goldenRatioInv) :
    atasDensity r < atasDensity goldenRatioInv := by
  -- Since r ≠ goldenRatioInv and r > 0, by atasDensity_eq_max_iff we have atasDensity r ≠ 1/16.
  have h_neq : atasDensity r ≠ 1 / 16 := by
    exact fun h => hne <| atasDensity_eq_max_iff r hr |>.1 h;
  exact lt_of_le_of_ne ( by simpa [ atasDensity_at_goldenRatioInv ] using atasDensity_le r hr.le ) fun h => h_neq <| by simpa [atasDensity_at_goldenRatioInv] using h;

/-! ## Symmetry of the density -/

/-
The Atas density satisfies p₂(r) = r⁻² · p₂(1/r) for r > 0.
-/
theorem atasDensity_symmetry (r : ℝ) (hr : r > 0) :
    atasDensity r = r⁻¹ ^ 2 * atasDensity (1 / r) := by
  unfold atasDensity;
  -- Simplify the right-hand side of the equation.
  field_simp
  ring_nf

/-! ## Relationship between golden ratio and its inverse -/

/-
(√5-1)/2 = ((1+√5)/2)⁻¹
-/
theorem goldenRatioInv_eq_inv_golden :
    goldenRatioInv = goldenRatio⁻¹ := by
  rw [goldenRatioInv_explicit]
  unfold goldenRatio
  rw [inv_eq_of_mul_eq_one_right]
  nlinarith [Real.sq_sqrt (show (0:ℝ) ≤ 5 by norm_num), Real.sqrt_nonneg 5]

/-- CJ-02: GUE mode is the golden ratio inverse. -/
def buchanan_gue_mode := @gue_mode_is_golden_ratio_inv

end MNZI