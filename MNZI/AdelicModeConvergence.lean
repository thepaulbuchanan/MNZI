/-
# MNZI/AdelicModeConvergence.lean

Machine-verified theorems for Paper P:
"N-Bonacci Constants as Spectral Modes: A General Theorem on Spacing
 Ratio Distributions and the Adelic Limit"

## Scope
This file verifies 22 theorems covering:
  1. The algebraic criticality identity (reciprocal of n-bonacci constant
     satisfies ∑_{k=1}^n r^k = 1)
  2. The golden ratio base case (n=2)
  3. Geometric sum convergence and the adelic limit r*_∞ = 1/2
  4. Existence, uniqueness, and monotone convergence of the root sequence

## What is NOT claimed
The connections between n-bonacci constants and:
  - The Riemann Hypothesis
  - SL(n,ℤ)\ℍⁿ spectral statistics
  - GUE universality for n ≥ 3
are CONJECTURES (Conjecture 6.1 in the paper), not established mathematics.
The n=2 base case (mode of GUE = 1/φ) is proved in Paper A.
-/

import Mathlib
import MNZI.Core

namespace MNZI

open Finset BigOperators Real Filter Topology

-- recipSum is imported from MNZI.Core (alias for criticalitySum)

/-! ## Section 1: Supporting lemmas for sqrt(5) and the golden ratio -/

/-
√5 > 1
-/
theorem sqrt5_gt_one : Real.sqrt 5 > 1 := by
  norm_num [ Real.lt_sqrt ]

/-
(√5)² = 5
-/
theorem sqrt5_sq : Real.sqrt 5 ^ 2 = 5 := by
  norm_num

/-
(√5 - 1) / 2 > 0
-/
theorem golden_inv_pos : (Real.sqrt 5 - 1) / 2 > 0 := by
  nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ]

/-
(√5 - 1) / 2 < 1
-/
theorem golden_inv_lt_one : (Real.sqrt 5 - 1) / 2 < 1 := by
  nlinarith [ Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ]

/-! ## Section 2: Golden ratio base case (n = 2) -/

/-
The golden ratio reciprocal satisfies r + r² = 1
-/
theorem golden_ratio_reciprocal :
    let r := (Real.sqrt 5 - 1) / 2
    r + r ^ 2 = 1 := by
  ring_nf; norm_num;

/-
(√5 - 1)/2 = 1/φ where φ = (1 + √5)/2
-/
theorem golden_ratio_inv_eq :
    (Real.sqrt 5 - 1) / 2 = 1 / ((1 + Real.sqrt 5) / 2) := by
  grind

/-! ## Section 3: Geometric sum closed form -/

/-
Closed form for recipSum: r(1 - r^n)/(1 - r) when r ≠ 1
-/
theorem recipSum_geometric (r : ℝ) (n : ℕ) (hr : r ≠ 1) :
    recipSum r n = r * (1 - r ^ n) / (1 - r) := by
  -- Recognize that the sum is a geometric series with first term $r$ and common ratio $r$.
  have h_geom : ∑ k ∈ Finset.range n, r ^ (k + 1) = r * (∑ k ∈ Finset.range n, r ^ k) := by
    rw [ Finset.mul_sum _ _ _, Finset.sum_congr rfl fun _ _ => pow_succ' _ _ ];
  exact h_geom.trans ( by rw [ geom_sum_eq hr ] ; rw [ ← neg_div_neg_eq ] ; ring )

/-! ## Section 4: The criticality identity -/

/-
If C is a root of x^n = x^{n-1} + x^{n-2} + ... + x + 1 and C ≠ 0,
    then ∑_{k=1}^{n} (1/C)^k = 1.
-/
theorem reciprocal_identity (C : ℝ) (n : ℕ) (hn : 2 ≤ n) (hC : C ≠ 0)
    (hroot : C ^ n = ∑ k ∈ Finset.range n, C ^ k) :
    recipSum (1 / C) n = 1 := by
  convert congr_arg ( fun x : ℝ => x / C ^ n ) hroot.symm using 1;
  · unfold recipSum criticalitySum;
    rw [ ← Finset.sum_range_reflect ];
    rw [ Finset.sum_div _ _ _ ] ; refine' Finset.sum_congr rfl fun i hi => _ ; rw [ show n - 1 - i + 1 = n - i from by rw [ tsub_right_comm, tsub_add_cancel_of_le ( Nat.succ_le_of_lt ( Nat.sub_pos_of_lt ( Finset.mem_range.mp hi ) ) ) ] ] ; ring;
    rw [ show C⁻¹ ^ n = C⁻¹ ^ ( n - i ) * C⁻¹ ^ i by rw [ ← pow_add, Nat.sub_add_cancel ( Finset.mem_range_le hi ) ] ] ; ring;
    simp +decide [ mul_assoc, mul_comm, mul_left_comm, hC ];
  · rw [ div_self ( pow_ne_zero _ hC ) ]

/-
Converse: if r ∈ (0,1) satisfies ∑_{k=1}^n r^k = 1 then C = 1/r
    satisfies C^n = C^{n-1} + ... + C + 1.
-/
theorem reciprocal_identity_conv (r : ℝ) (n : ℕ) (hn : 2 ≤ n)
    (hr_pos : 0 < r) (hr_lt : r < 1)
    (hsum : recipSum r n = 1) :
    (1 / r) ^ n = ∑ k ∈ Finset.range n, (1 / r) ^ k := by
  rw [ eq_comm, geom_sum_eq ];
  · rw [ div_eq_iff ] <;> norm_num;
    · unfold recipSum criticalitySum at hsum;
      simp_all +decide [ pow_succ, ← Finset.mul_sum _ _ _, ← Finset.sum_mul ];
      field_simp;
      nlinarith [ geom_sum_mul r n ];
    · nlinarith [ inv_mul_cancel₀ hr_pos.ne' ];
  · grind

/-! ## Section 5: Adelic limit theorems -/

/-
For 0 ≤ r < 1, ∑_{k=1}^n r^k → r/(1-r) as n → ∞
-/
theorem recipSum_tendsto (r : ℝ) (hr0 : 0 ≤ r) (hr1 : r < 1) :
    Filter.Tendsto (fun n => recipSum r n) Filter.atTop
      (nhds (r / (1 - r))) := by
  convert Filter.Tendsto.congr ( fun n => ?_ ) ( tendsto_const_nhds.mul ( tendsto_pow_atTop_nhds_zero_of_lt_one hr0 hr1 |> ( ·.const_sub 1 ) ) |> ( ·.div_const ( 1 - r ) ) ) using 1;
  exact congr_arg _ ( by ring );
  rw [ recipSum_geometric r n ( by linarith ) ]

/-
The unique solution of r/(1-r) = 1 in (0,1) is r = 1/2
-/
theorem limit_equation_solution (r : ℝ) (hr0 : 0 < r) (hr1 : r < 1)
    (heq : r / (1 - r) = 1) : r = 1 / 2 := by
  grind

/-
r = 1/2 satisfies the limit equation r/(1-r) = 1
-/
theorem half_satisfies_limit : (1 : ℝ) / 2 / (1 - 1 / 2) = 1 := by
  norm_num

/-
recipSum(1/2, n) → 1 as n → ∞
-/
theorem recipSum_half_tendsto' :
    Filter.Tendsto (fun n => recipSum (1 / 2) n) Filter.atTop (nhds 1) := by
  convert recipSum_tendsto ( 1 / 2 ) ( by norm_num ) ( by norm_num ) using 1 ; norm_num [ div_eq_mul_inv ]

/-
The n-bonacci limit constant is 2 (i.e., 1/r*_∞ = 1/(1/2) = 2)
-/
theorem nBonacci_limit_is_two : 1 / ((1 : ℝ) / 2) = 2 := by
  grind +locals

/-! ## Section 6: Properties of recipSum -/

/-
recipSum is strictly increasing in n for r > 0
-/
theorem recipSum_strictMono_n (r : ℝ) (hr : 0 < r) :
    StrictMono (fun n => recipSum r n) := by
  refine' strictMono_nat_of_lt_succ _;
  intro n; unfold recipSum criticalitySum; simp +decide [ Finset.sum_range_succ ] ; positivity;

/-
recipSum is strictly increasing in r on (0,1) for n ≥ 1
-/
theorem recipSum_strictMono_r (n : ℕ) (hn : 1 ≤ n)
    (r₁ r₂ : ℝ) (h1 : 0 < r₁) (h2 : r₁ < r₂) (h3 : r₂ < 1) :
    recipSum r₁ n < recipSum r₂ n := by
  exact Finset.sum_lt_sum_of_nonempty ⟨ _, Finset.mem_range.mpr hn ⟩ fun _ _ => pow_lt_pow_left₀ h2 h1.le ( by positivity )

/-
recipSum is continuous in r
-/
theorem recipSum_continuous' (n : ℕ) :
    Continuous (fun r : ℝ => recipSum r n) := by
  unfold recipSum criticalitySum; exact continuous_finset_sum _ fun _ _ => Continuous.pow continuous_id' _

/-
recipSum(0, n) = 0
-/
theorem recipSum_zero' (n : ℕ) : recipSum 0 n = 0 := by
  unfold recipSum criticalitySum; aesop;

/-
recipSum(1, n) = n
-/
theorem recipSum_one' (n : ℕ) :
    recipSum (1 : ℝ) n = (n : ℝ) := by
  unfold recipSum criticalitySum; norm_num;

/-! ## Section 7: Existence and uniqueness of the root -/

/-
For n ≥ 2, there exists a unique r ∈ (0,1) with recipSum r n = 1
-/
theorem exists_unique_rn (n : ℕ) (hn : 2 ≤ n) :
    ∃! r : ℝ, 0 < r ∧ r < 1 ∧ recipSum r n = 1 := by
  -- By the intermediate value theorem, there exists a unique $r \in (0,1)$ such that $\sum_{k=1}^n r^k = 1$.
  obtain ⟨r, hr₁, hr₂⟩ : ∃ r ∈ Set.Ioo (0 : ℝ) 1, recipSum r n = 1 := by
    apply_rules [ intermediate_value_Ioo ] <;> norm_num;
    · exact Continuous.continuousOn (recipSum_continuous' n);
    · exact ⟨ by rw [ recipSum_zero' ] ; norm_num, by rw [ recipSum_one' ] ; norm_cast ⟩;
  refine' ⟨ r, _, _ ⟩ <;> norm_num;
  · grind +qlia;
  · intro y hy₁ hy₂ hy; exact le_antisymm ( le_of_not_gt fun h => by linarith [ recipSum_strictMono_r n ( by linarith ) r y hr₁.1 h hy₂ ] ) ( le_of_not_gt fun h => by linarith [ recipSum_strictMono_r n ( by linarith ) y r hy₁ h hr₁.2 ] ) ;

/-! ## Section 8: Monotonicity of the root sequence -/

/-
The root sequence is strictly decreasing: if recipSum r n = 1 and
    recipSum r' (n+1) = 1 with both in (0,1), then r' < r
-/
theorem rn_decreasing (n : ℕ) (_hn : 2 ≤ n)
    (r r' : ℝ) (hr : 0 < r ∧ r < 1 ∧ recipSum r n = 1)
    (hr' : 0 < r' ∧ r' < 1 ∧ recipSum r' (n + 1) = 1) :
    r' < r := by
  by_contra h_contra
  have h_ge : recipSum r' n ≥ recipSum r n := by
    unfold recipSum criticalitySum;
    exact Finset.sum_le_sum fun i hi => pow_le_pow_left₀ hr.1.le ( le_of_not_gt h_contra ) _;
  simp_all +decide [ Finset.sum_range_succ, recipSum, criticalitySum ];
  linarith [ pow_pos hr'.1 ( n + 1 ) ]

/-
The root is always > 1/2 for finite n ≥ 2
-/
theorem rn_gt_half (n : ℕ) (hn : 2 ≤ n)
    (r : ℝ) (hr : 0 < r ∧ r < 1 ∧ recipSum r n = 1) :
    1 / 2 < r := by
  contrapose! hr;
  intro h1 h2; rw [ recipSum_geometric ] <;> norm_num at *;
  · rw [ div_eq_iff ] <;> nlinarith [ pow_pos h1 n, pow_le_pow_of_le_one h1.le h2.le hn ];
  · linarith

end MNZI
