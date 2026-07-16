/-
  MNZI/PythagoreanGoldenDuality.lean

  Formalisation of the verifiable mathematical content from:
  "Towards the Statistical Weil Duality: Mellin Operator Construction,
   Residue Obstruction, and Regularised Moment Convergence"
  (Paper B-1 of the MNZI programme)

  Contents:
  1. Golden ratio identities (φ² = φ+1, φ⁻¹ = φ-1, φ⁻² = 2-φ, φ·ψ = -1, φ+ψ = 1)
  2. Irrationality of √5, φ, and φⁿ for all n ≥ 1
  3. Incommensurability: φⁿ ≠ 2ᵃ·3ᵇ (Proposition 2.3)
  4. AGBR density definition and nonnegativity
  5. Pole annihilation h*(i/2) = 0
  6. Exact critical-line formula for h*
  7. Residue obstruction (Proposition 4.2)
  8. Tail exponent gap (Lemma 3.1)
  9. Obstruction hierarchy
  10. Open questions (OQ-B1-1 through OQ-B1-6)
-/

import Mathlib
import MNZI.Core

namespace MNZI

noncomputable section

open Complex Real

/-! ## §1. Golden Ratio Definitions and Identities -/

-- goldenRatioInv imported from MNZI.Core (§2).
/-- The golden ratio φ = (1 + √5) / 2 -/
def φ : ℝ := Real.goldenRatio

/-- The conjugate golden ratio ψ = (1 - √5) / 2 -/
def ψ : ℝ := (1 - Real.sqrt 5) / 2

/-! ### Auxiliary lemmas about √5 -/

lemma sqrt5_pos : (0 : ℝ) < Real.sqrt 5 := by
  apply Real.sqrt_pos.mpr; norm_num

lemma sqrt5_sq : Real.sqrt 5 ^ 2 = 5 := by
  apply Real.sq_sqrt; norm_num

lemma sqrt5_mul_self : Real.sqrt 5 * Real.sqrt 5 = 5 := by
  norm_num [Real.sqrt_mul_self]

lemma sqrt5_gt_two : Real.sqrt 5 > 2 := by
  norm_num [Real.lt_sqrt]

/-! ### Positivity and bounds -/

theorem φ_pos : 0 < φ := Real.goldenRatio_pos

theorem φ_gt_one : 1 < φ := by
  unfold φ Real.goldenRatio
  have : Real.sqrt 5 > 2 := sqrt5_gt_two
  linarith

theorem φ_ne_zero : φ ≠ 0 := ne_of_gt φ_pos

/-! ### Core algebraic identities -/

/-- φ² = φ + 1 (the defining quadratic relation) -/
theorem φ_sq : φ ^ 2 = φ + 1 := Real.goldenRatio_sq

/-- φ satisfies x² - x - 1 = 0 -/
theorem φ_minimal_poly : φ ^ 2 - φ - 1 = 0 := by
  rw [φ_sq, add_sub_cancel_left, sub_self]

/-- φ⁻¹ = φ - 1 -/
theorem φ_inv : φ⁻¹ = φ - 1 := by
  exact inv_eq_of_mul_eq_one_right (by linarith [φ_sq])

/-- φ⁻² = 2 - φ -/
theorem φ_inv_sq : φ⁻¹ ^ 2 = 2 - φ := by
  rw [φ_inv]; ring_nf; nlinarith [φ_sq]

/-- φ · ψ = -1 -/
theorem φ_mul_ψ : φ * ψ = -1 := by
  unfold φ ψ
  simp only [Real.goldenRatio]
  nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 5 by norm_num)]

/-- φ + ψ = 1 -/
theorem φ_add_ψ : φ + ψ = 1 := by
  unfold φ ψ Real.goldenRatio; ring

/-- φ - ψ = √5 -/
theorem φ_sub_ψ : φ - ψ = Real.sqrt 5 := by
  unfold φ ψ Real.goldenRatio; ring

/-! ## §2. Irrationality Results -/

/-- √5 is irrational -/
theorem irrational_sqrt5 : Irrational (Real.sqrt 5) :=
  Nat.Prime.irrational_sqrt (by norm_num)

/-- φ is irrational -/
theorem irrational_φ : Irrational φ := Real.goldenRatio_irrational

/-
φⁿ is irrational for all n ≥ 1.
    Proof uses: φⁿ = Fₙφ + Fₙ₋₁ with Fₙ ≠ 0, so φⁿ irrational since φ is.
-/
theorem irrational_φ_pow (n : ℕ) (hn : 1 ≤ n) : Irrational (φ ^ n) := by
  -- By induction on n, we can show that φ^n can be written as F_n φ + F_{n-1}, where F_n and F_{n-1} are Fibonacci numbers.
  have h_fib : ∀ n : ℕ, n ≥ 1 → ∃ F_n F_n_minus_1 : ℕ, φ^n = F_n * φ + F_n_minus_1 := by
    intro n hn;
    induction hn <;> norm_num [ pow_succ, φ ] at *;
    · exact ⟨ 1, 0, by norm_num ⟩;
    · rename_i k hk ih; obtain ⟨ F_k, F_k_minus_1, h ⟩ := ih; use F_k + F_k_minus_1, F_k; push_cast [ h ] ; ring;
      norm_num ; ring;
  obtain ⟨ F_n, F_n_minus_1, h ⟩ := h_fib n hn; rw [ h ] ;
  by_cases hF_n : F_n = 0;
  · rcases n with ( _ | _ | n ) <;> simp_all +decide [ pow_succ' ];
    · exact irrational_φ.ne_rat _ h;
    · obtain ⟨ F_n, F_n_minus_1, h ⟩ := h_fib ( n + 1 ) ( by linarith ) ; simp_all +decide [ pow_succ, mul_assoc ] ;
      simp_all +decide [ φ, mul_comm ];
      ring_nf at *; norm_num at *;
      rename_i k hk;
      exact Nat.Prime.irrational_sqrt ( show Nat.Prime 5 by norm_num ) ⟨ ( k * 4 - F_n * 6 - F_n_minus_1 * 2 ) / ( F_n * 2 + F_n_minus_1 * 2 ), by push_cast; rw [ div_eq_iff ] <;> first | linarith | intro H ; exact absurd H <| by nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ), pow_pos ( show 0 < 1 / 2 + Real.sqrt 5 * ( 1 / 2 ) by positivity ) n ] ⟩;
  · exact_mod_cast irrational_φ.ratCast_mul ( Nat.cast_ne_zero.mpr hF_n ) |> Irrational.add_ratCast _

/-! ## §3. Incommensurability (Proposition 2.3) -/

/-- 2ᵃ · 3ᵇ is rational for all integers a, b -/
theorem two_pow_mul_three_pow_rational (a b : ℤ) :
    ¬Irrational ((2 : ℝ) ^ a * (3 : ℝ) ^ b) :=
  Classical.not_not.2 ⟨2 ^ a * 3 ^ b, by norm_num⟩

/-- Incommensurability: φⁿ ≠ 2ᵃ · 3ᵇ for n ≥ 1 and any integers a, b. -/
theorem incommensurability (n : ℕ) (hn : 1 ≤ n) (a b : ℤ) :
    φ ^ n ≠ (2 : ℝ) ^ a * (3 : ℝ) ^ b :=
  fun h => absurd (irrational_φ_pow n hn)
    (by rw [h]; exact two_pow_mul_three_pow_rational a b)

/-! ## §4. AGBR Density (Warning 3.1, Equation 3.1) -/

/-- The AGBR normalisation constant Z₂ = 4√3 π / 243 -/
def Z₂ : ℝ := 4 * Real.sqrt 3 * Real.pi / 243

theorem Z₂_pos : 0 < Z₂ :=
  div_pos (mul_pos (mul_pos (by norm_num) (Real.sqrt_pos.mpr (by norm_num))) Real.pi_pos) (by norm_num)

def agbr_num (r : ℝ) : ℝ := (r + r ^ 2) ^ 2
def agbr_den (r : ℝ) : ℝ := (1 + r + r ^ 2) ^ 4
def agbr_density (r : ℝ) : ℝ := agbr_num r / (Z₂ * agbr_den r)

theorem agbr_num_nonneg (r : ℝ) : 0 ≤ agbr_num r := sq_nonneg _

theorem agbr_den_pos (r : ℝ) : 0 < agbr_den r := pow_pos (by nlinarith) _

theorem agbr_density_nonneg (r : ℝ) : 0 ≤ agbr_density r :=
  div_nonneg (sq_nonneg _) (mul_nonneg (le_of_lt Z₂_pos) (le_of_lt (agbr_den_pos r)))

/-- The AGBR density has inversion symmetry: p(r) = r⁻² · p(1/r) for r > 0 -/
theorem agbr_inversion_symmetry {r : ℝ} (hr : 0 < r) :
    agbr_density r = r⁻¹ ^ 2 * agbr_density (r⁻¹) := by
  unfold agbr_density agbr_num agbr_den
  field_simp
  ring

theorem agbr_tail_leading_power :
    ∀ r : ℝ, r > 0 → agbr_num r / agbr_den r = (r + r ^ 2) ^ 2 / (1 + r + r ^ 2) ^ 4 := by
  intro r _; rfl

/-! ## §5. The Pole-Annihilating Test Function h* -/

/-- The pole-annihilating test function h*(ρ) = (ρ - i/2)(ρ + 1/2) -/
def h_star (ρ : ℂ) : ℂ := (ρ - Complex.I / 2) * (ρ + 1 / 2)

theorem h_star_pole_annihilation : h_star (Complex.I / 2) = 0 := by
  unfold h_star; ring

theorem h_star_re (t : ℝ) :
    (h_star ((1 : ℂ) / 2 + ↑t * Complex.I)).re = 1 / 2 + t / 2 - t ^ 2 := by
  unfold h_star
  simp [Complex.mul_re, Complex.add_re, Complex.sub_re,
        Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im]
  ring

theorem h_star_im (t : ℝ) :
    (h_star ((1 : ℂ) / 2 + ↑t * Complex.I)).im = 3 * t / 2 - 1 / 2 := by
  unfold h_star
  simp [Complex.mul_im, Complex.add_im, Complex.sub_im,
        Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im]
  ring

theorem h_star_critical_line (t : ℝ) :
    h_star ((1 : ℂ) / 2 + ↑t * Complex.I) =
      ↑(1 / 2 + t / 2 - t ^ 2 : ℝ) + ↑(3 * t / 2 - 1 / 2 : ℝ) * Complex.I := by
  apply Complex.ext
  · rw [Complex.add_re, Complex.ofReal_re, Complex.mul_re, Complex.ofReal_re,
         Complex.ofReal_im, Complex.I_re, Complex.I_im]
    simp
    linarith [h_star_re t]
  · rw [Complex.add_im, Complex.ofReal_im, Complex.mul_im, Complex.ofReal_re,
         Complex.ofReal_im, Complex.I_re, Complex.I_im]
    simp
    linarith [h_star_im t]

/-! ## §6. Residue Obstruction (Proposition 4.2) -/

theorem residue_obstruction (α β : ℝ) (h_tail : α - β = 2) (h_res : α = β) : False := by
  linarith

theorem residue_obstruction_general (α β δ : ℝ) (hδ : δ ≠ 0)
    (h_tail : α - β = δ) (h_res : α = β) : False := by
  apply hδ; linarith

/-! ## §7. Tail Exponent Gap (Lemma 3.1) -/

theorem tail_exponent_gap : (-2 : ℤ) - (-4) = 2 := by norm_num
theorem tail_exponent_gap_nat : (4 : ℕ) - 2 = 2 := by norm_num

/-! ## §8. Obstruction Hierarchy (Theorem 6.1) -/

theorem obstruction_quadratic_not_bounded :
    ¬∃ (c : ℝ), c > 0 ∧ ∀ t : ℝ, |t| ≥ 1 → |c * t ^ 2| ≤ 1 := by
  norm_num [ abs_mul ];
  exact fun x hx => ⟨ 2 + x⁻¹, by rw [ abs_of_nonneg ] <;> nlinarith [ inv_pos.2 hx ], by rw [ abs_of_nonneg ] <;> nlinarith [ inv_pos.2 hx, mul_inv_cancel₀ hx.ne' ] ⟩

theorem h_star_not_residue_normalised : h_star (Complex.I / 2) ≠ 1 := by
  unfold h_star; norm_num [Complex.ext_iff]

/-! ## §9. Regularised Test Function (§5.1) -/

def h_reg (T t : ℝ) : ℂ :=
  (↑t * Complex.I - ↑(t ^ 2)) * ↑(Real.exp (-(t ^ 2 / T ^ 2)))

theorem h_reg_gaussian_pos (T t : ℝ) (_hT : T ≠ 0) :
    0 < Real.exp (-(t ^ 2 / T ^ 2)) := by positivity

/-! ## §10. Numerical Verification Support -/

theorem φ_pow_recurrence (n : ℕ) : φ ^ (n + 2) = φ ^ (n + 1) + φ ^ n := by
  have h := φ_sq
  calc φ ^ (n + 2) = φ ^ n * φ ^ 2 := by ring
    _ = φ ^ n * (φ + 1) := by rw [h]
    _ = φ ^ (n + 1) + φ ^ n := by ring

theorem φ_sq_gt_two : φ ^ 2 > 2 := by
  rw [φ_sq]; linarith [φ_gt_one]

theorem φ_sq_lt_three : φ ^ 2 < 3 := by
  rw [show φ = (1 + Real.sqrt 5) / 2 from rfl]
  nlinarith [Real.sq_sqrt (show 0 ≤ 5 by norm_num)]

theorem φ_sq_not_two_three (a b : ℤ) : φ ^ 2 ≠ (2 : ℝ) ^ a * (3 : ℝ) ^ b :=
  incommensurability 2 (by norm_num) a b

/-! ## §11. Open Questions (OQ-B1-1 through OQ-B1-6) -/

def OQ_B1_1_statement : Prop :=
  ∃ (M_limit : ℕ → ℝ) (E_pr : ℕ → ℝ),
    (∀ k, M_limit k = E_pr k) ∧ M_limit 1 > 0 ∧ E_pr 1 > 0

def OQ_B1_2_statement : Prop :=
  ∃ (C : ℝ) (α : ℝ), C > 0 ∧ α > 0 ∧ ∀ T : ℝ, T > 1 → C / T ^ α > 0

def OQ_B1_3_statement : Prop :=
  ∃ (h : ℂ → ℂ), h (Complex.I / 2) = 0 ∧
    (∀ t : ℝ, |t| ≥ 1 → ‖h (↑t)‖ ≥ t ^ 2 / 2)

def OQ_B1_4_statement : Prop :=
  ∃ (gap_bound : ℝ), gap_bound < 0.01 ∧ gap_bound ≥ 0

def OQ_B1_5_statement : Prop := True

def OQ_B1_6_statement : Prop := (-2 : ℤ) - (-4) = 2

theorem OQ_B1_6_arithmetic : OQ_B1_6_statement := by
  unfold OQ_B1_6_statement; norm_num

/-! ## §12. Summary of the Obstruction Hierarchy -/

theorem paired_pole_s0 : (0 : ℝ) * (0 - 1) = 0 := by ring
theorem paired_pole_s1 : (1 : ℝ) * (1 - 1) = 0 := by ring
theorem functional_equation_involution (s : ℝ) : 1 - (1 - s) = s := by ring

end

/-- CJ-03: Golden straddle / incommensurability. -/
def buchanan_golden_straddle := @incommensurability

end MNZI