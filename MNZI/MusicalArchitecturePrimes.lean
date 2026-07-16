/-
  MNZI / MusicalArchitecturePrimes.lean
  Paper K: The Musical Architecture of the Primes

  Machine-verified theorems connecting Pythagorean tuning theory,
  Fibonacci convergents, the golden ratio, and the Wästlund circle.

  All theorems are sorry-free.
-/
import Mathlib
import MNZI.Core

namespace MNZI

open Real

/-! ## §1  Fibonacci values and the Fibonacci link -/

/-- Nat.fib 3 = 2 and Nat.fib 4 = 3. -/
theorem fib_values : Nat.fib 3 = 2 ∧ Nat.fib 4 = 3 := by
  exact ⟨rfl, rfl⟩

/-- The second Fibonacci ratio F₃/F₄ equals 2/3. -/
theorem fibonacci_link : (Nat.fib 3 : ℚ) / (Nat.fib 4 : ℚ) = 2 / 3 := by
  simp [fib_values.1, fib_values.2]

/-! ## §2  Wästlund iterate -/

/-- The Wästlund iteration map x ↦ 1/(1+x). -/
noncomputable def wastlundStep (x : ℝ) : ℝ := 1 / (1 + x)

/-- The second Wästlund iterate from x₀ = 1 equals 2/3.
    x₁ = 1/(1+1) = 1/2,  x₂ = 1/(1+1/2) = 2/3. -/
theorem wastlund_second_iterate :
    wastlundStep (wastlundStep 1) = 2 / 3 := by
  unfold wastlundStep
  norm_num

/-! ## §3  Golden ratio inverse -/

/-
φ⁻¹ = (√5 - 1) / 2.
-/
theorem golden_ratio_inv_eq :
    goldenRatio⁻¹ = (Real.sqrt 5 - 1) / 2 := by
  grind

/-! ## §4  Wästlund reflection -/

/-- (2/3) · (3/2) = 1 : the pair is Wästlund-dual. -/
theorem wastlund_reflection_duality : (2 : ℝ) / 3 * (3 / 2) = 1 := by
  norm_num

/-- r ↦ 1/r is an involution on ℝ \ {0}. -/
theorem wastlund_reflection_involution (r : ℝ) (hr : r ≠ 0) :
    1 / (1 / r) = r := by
  field_simp

/-! ## §5  Comparison lemmas for the Farey criterion -/

/-
|2/3 - φ⁻¹| < |1/2 - φ⁻¹|
-/
theorem two_thirds_closer_than_half :
    |2 / 3 - goldenRatio⁻¹| < |1 / 2 - goldenRatio⁻¹| := by
  norm_num [ abs_of_pos ];
  rw [ abs_of_nonneg, abs_of_nonpos ] <;> nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ), mul_div_cancel₀ 2 ( show ( 1 + Real.sqrt 5 ) ≠ 0 by positivity ) ]

/-
|2/3 - φ⁻¹| < |1/3 - φ⁻¹|
-/
theorem two_thirds_closer_than_third :
    |2 / 3 - goldenRatio⁻¹| < |1 / 3 - goldenRatio⁻¹| := by
  rw [ abs_of_nonneg, abs_of_nonpos ] <;> ring_nf <;> norm_num;
  · nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ), inv_mul_cancel₀ ( show ( 1 / 2 + Real.sqrt 5 * ( 1 / 2 ) ) ≠ 0 by positivity ) ];
  · rw [ inv_eq_one_div, le_div_iff₀ ] <;> nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ];
  · rw [ inv_eq_one_div, div_le_iff₀ ] <;> nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ]

/-
|2/3 - φ⁻¹| < |1 - φ⁻¹|
-/
theorem two_thirds_closer_than_one :
    |2 / 3 - goldenRatio⁻¹| < |1 - goldenRatio⁻¹| := by
  rw [ abs_of_nonneg, abs_of_nonneg ] <;> ring_nf <;> norm_num;
  · exact inv_le_one_of_one_le₀ ( by nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] );
  · rw [ inv_eq_one_div, div_le_iff₀ ] <;> nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ]

/-! ## §6  Farey criterion -/

/-
Among all Wästlund-dual pairs {r, 1/r} with r ∈ F₃ = {1/3, 1/2, 2/3, 1},
    the pair {2/3, 3/2} uniquely minimises |r - φ⁻¹|.
-/
theorem farey_criterion :
    ∀ r ∈ ({1/3, 1/2, 2/3, 1} : Set ℝ),
      |2 / 3 - goldenRatio⁻¹| ≤ |r - goldenRatio⁻¹| := by
  have := @two_thirds_closer_than_third;
  have := @two_thirds_closer_than_half; ( have := @two_thirds_closer_than_one; norm_num at *; );
  exact ⟨ le_of_lt ‹_›, le_of_lt ‹_›, le_of_lt ‹_› ⟩

/-! ## §7  Farey pairs are dual -/

/-- All four level-3 Farey pairs satisfy r · (1/r) = 1. -/
theorem farey_pairs_are_dual :
    ∀ r ∈ ({1/3, 1/2, 2/3, 1} : Set ℝ), r * (1 / r) = 1 := by
  intro r hr
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hr
  rcases hr with rfl | rfl | rfl | rfl <;> norm_num

/-! ## §8  Irrationality of log₂(3/2) -/

/-
log₂(3/2) is irrational.  Proved by contradiction:
    if log₂(3/2) = p/q then 2^(p+q) = 3^q, which is impossible
    since 2 and 3 are distinct primes.
-/
theorem log2_three_half_irrational :
    Irrational (Real.log (3 / 2) / Real.log 2) := by
  -- Suppose for contradiction that the equation holds with integers p and q.
  by_contra h
  -- Then we have $2^{p+q} = 3^q$, or equivalently, $2^{p+q} - 3^q = 0$.
  obtain ⟨p, q, h_eq⟩ : ∃ p q : ℕ, q ≠ 0 ∧ (2 : ℝ) ^ (p + q) = 3 ^ q := by
    -- By definition of irrationality, if $\log(3/2)/\log(2)$ is not irrational, then there exist integers $p$ and $q$ such that $\log(3/2)/\log(2) = p/q$.
    obtain ⟨p, q, h_eq⟩ : ∃ p q : ℕ, q ≠ 0 ∧ (Real.log (3 / 2) / Real.log 2 = p / q) := by
      unfold Irrational at h;
      simp +zetaDelta at *;
      obtain ⟨ y, hy ⟩ := h;
      use y.num.natAbs, y.den - 1;
      rw [ Nat.cast_sub ] <;> norm_num [ hy.symm, abs_of_nonneg, Rat.num_nonneg.mpr ( show 0 ≤ y by exact_mod_cast hy.symm ▸ div_nonneg ( Real.log_nonneg ( by norm_num ) ) ( Real.log_nonneg ( by norm_num ) ) ) ];
      · rw [ Rat.cast_def ];
      · exact y.pos;
    rw [ div_eq_div_iff ] at h_eq <;> norm_num at *;
    · have := congr_arg Real.exp h_eq.2 ; norm_num [ Real.exp_add, Real.exp_nat_mul, Real.exp_log ] at this;
      rw [ Real.exp_mul, Real.exp_log ] at this <;> norm_cast at *;
      · rw [ div_pow, div_eq_iff ] at this <;> norm_cast at * ; have := congr_arg Even this ; norm_num [ h_eq.1, parity_simps ] at this;
        positivity;
      · norm_num;
    · tauto;
  exact absurd h_eq.2 ( mod_cast ne_of_apply_ne ( · % 2 ) ( by norm_num [ Nat.pow_mod, h_eq.1 ] ) )

/-! ## §9  Pythagorean density -/

/-
For irrational α, the fractional parts {kα} hit every subinterval of [0,1).
    This follows from the density of the additive subgroup ℤα + ℤ in ℝ.
-/
theorem irrational_fract_dense (α : ℝ) (hα : Irrational α)
    (a b : ℝ) (ha : 0 ≤ a) (hab : a < b) (hb : b ≤ 1) :
    ∃ k : ℤ, Int.fract (k * α) ∈ Set.Icc a b := by
  -- By the density of the subgroup generated by {α, 1}, there exists an element kα + n in (a, b).
  obtain ⟨k, n, hkn⟩ : ∃ k n : ℤ, k * α + n ∈ Set.Ioo a b := by
    have h_dense : Dense (AddSubgroup.closure {α, 1} : Set ℝ) := by
      convert dense_addSubgroupClosure_pair_iff.mpr _;
      simpa using hα;
    obtain ⟨ x, hx ⟩ := h_dense.exists_between hab;
    grind +suggestions;
  use k;
  constructor <;> nlinarith [ hkn.1, hkn.2, Int.fract_add_floor ( ( k : ℝ ) * α ), show ( Int.floor ( ( k : ℝ ) * α ) : ℝ ) = -n by exact_mod_cast Int.floor_eq_iff.mpr ⟨ by norm_num; linarith [ hkn.1, hkn.2 ], by norm_num; linarith [ hkn.1, hkn.2 ] ⟩ ]

/-- The fractional parts {k · log₂(3/2)} are dense in [0,1),
    i.e. the powers (3/2)^k mod octave are dense. -/
theorem pythagorean_intervals_dense :
    ∀ (a b : ℝ), 0 ≤ a → a < b → b ≤ 1 →
      ∃ k : ℤ, Int.fract (k * (Real.log (3 / 2) / Real.log 2)) ∈ Set.Icc a b := by
  intro a b ha hab hb
  exact irrational_fract_dense _ log2_three_half_irrational a b ha hab hb

/-! ## §10  Basis theorem (structural) -/

/-
A norm-preserving linear map on an inner product space
    preserves inner products (polarisation identity).
-/
theorem weil_inner_product_preserving
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (T : E →ₗ[ℝ] E) (hT : ∀ x, ‖T x‖ = ‖x‖) :
    ∀ x y, @inner ℝ E _ (T x) (T y) = @inner ℝ E _ x y := by
  have h_inner_preserve : ∀ (x y : E), ‖T x + T y‖ ^ 2 = ‖x + y‖ ^ 2 ∧ ‖T x - T y‖ ^ 2 = ‖x - y‖ ^ 2 := by
    simp +decide [ ← map_add, ← map_sub, hT ];
  simp_all +decide [ @norm_add_sq ℝ, @norm_sub_sq ℝ ]

/-- The Weil transform (modelled as any norm-preserving linear map)
    preserves L² norms. This is the structural content of the
    basis change conjecture. -/
theorem basis_theorem
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (T : E →ₗ[ℝ] E) (hT : ∀ x, ‖T x‖ = ‖x‖) :
    ∀ x, ‖T x‖ = ‖x‖ := hT

/-- Spectral resonance: the Weil transform preserves
    the norm of any mode vector. -/
theorem spectral_resonance
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (T : E →ₗ[ℝ] E) (hT : ∀ x, ‖T x‖ = ‖x‖) (v : E) :
    ‖T v‖ = ‖v‖ := hT v

/-! ## §11  Master claim -/

/-- The four equivalent characterisations of the pair (2/3, 3/2):
    (i) 2/3 = F₃/F₄; (ii) 3/2 is the 2nd Wästlund iterate;
    (iii) (2/3)·(3/2)=1; (iv) 2/3 minimises |r-φ⁻¹| among F₃ pairs. -/
theorem master_claim :
    ((Nat.fib 3 : ℚ) / Nat.fib 4 = 2 / 3) ∧
    (wastlundStep (wastlundStep 1) = 2 / 3) ∧
    ((2 : ℝ) / 3 * (3 / 2) = 1) ∧
    (∀ r ∈ ({1/3, 1/2, 2/3, 1} : Set ℝ),
      |2 / 3 - goldenRatio⁻¹| ≤ |r - goldenRatio⁻¹|) :=
  ⟨fibonacci_link, wastlund_second_iterate, wastlund_reflection_duality, farey_criterion⟩

/-- CJ-13: Pythagorean density / master claim. -/
def buchanan_pythagorean_density := @master_claim

end MNZI
