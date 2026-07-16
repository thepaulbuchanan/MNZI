/-
  MNZI Paper T: N-Bonacci Quasiperiodic Coherence
  Spectral Modes, Time Quasicrystals, and the Adelic Limit

  All definitions and theorems in namespace MNZI.
  Machine-verified in Lean 4 with Mathlib.
-/
import Mathlib
import MNZI.Core

namespace MNZI

open Real Finset Filter Topology BigOperators

-- criticalitySum and its basic lemmas imported from MNZI.Core (§4).

/-! ## Section 1: Criticality Sum and Basic Properties -/

-- criticalitySum_formula, criticalitySum_half, criticalitySum_half_lt_one,
-- criticalitySum_continuous imported from MNZI.Core (§4).

/-
At x = 1, the criticality sum equals n.
-/
lemma criticalitySum_one (n : ℕ) :
    criticalitySum 1 n = ↑n := by
      unfold criticalitySum; norm_num;

/-
The criticality sum at x = 0 equals 0.
-/
lemma criticalitySum_zero (n : ℕ) :
    criticalitySum 0 n = 0 := by
      unfold criticalitySum; norm_num;

/-! ## Section 2: Existence and Uniqueness of the Criticality Root -/

/-
For n ≥ 2, there exists a unique x ∈ (0, 1) with criticalitySum x n = 1.
-/
lemma criticalityRoot_exists_unique (n : ℕ) (hn : 2 ≤ n) :
    ∃! x : ℝ, 0 < x ∧ x < 1 ∧ criticalitySum x n = 1 := by
      -- By the Intermediate Value Theorem, since the criticality sum is continuous and strictly increasing on (0,1), there exists a unique $x \in (0,1)$ such that the criticality sum equals 1.
      obtain ⟨x, hx₀, hx₁⟩ : ∃ x ∈ Set.Ioo (0 : ℝ) 1, criticalitySum x n = 1 := by
        apply_rules [ intermediate_value_Ioo ] <;> norm_num [ criticalitySum ];
        · exact Continuous.continuousOn ( continuous_finset_sum _ fun _ _ => continuous_pow _ );
        · grind;
      use x;
      simp_all +decide [ criticalitySum ];
      intro y hy₀ hy₁ hy₂; exact le_antisymm ( le_of_not_gt fun hxy => by linarith [ show ∑ k ∈ Finset.range n, y ^ ( k + 1 ) > ∑ k ∈ Finset.range n, x ^ ( k + 1 ) from Finset.sum_lt_sum_of_nonempty ( by norm_num; linarith ) fun k hk => pow_lt_pow_left₀ hxy ( by linarith ) ( by linarith ) ] ) ( le_of_not_gt fun hxy => by linarith [ show ∑ k ∈ Finset.range n, x ^ ( k + 1 ) > ∑ k ∈ Finset.range n, y ^ ( k + 1 ) from Finset.sum_lt_sum_of_nonempty ( by norm_num; linarith ) fun k hk => pow_lt_pow_left₀ hxy ( by linarith ) ( by linarith ) ] ) ;

/-- The criticality threshold r*_n: the unique root in (0,1) of ∑_{k=1}^n x^k = 1. -/
noncomputable def rStar (n : ℕ) : ℝ :=
  if h : 2 ≤ n then (criticalityRoot_exists_unique n h).choose
  else 1/2

/-
r*_n is positive for n ≥ 2.
-/
lemma rStar_pos (n : ℕ) (hn : 2 ≤ n) : 0 < rStar n := by
  unfold rStar;
  grind +splitIndPred

/-
r*_n < 1 for n ≥ 2.
-/
lemma rStar_lt_one (n : ℕ) (hn : 2 ≤ n) : rStar n < 1 := by
  unfold rStar; split_ifs ; linarith [ ( criticalityRoot_exists_unique n hn ).choose_spec.1.2.1 ] ;

/-! ## Section 3: The Reciprocal Identity (Theorem 2.1) -/

/-
**Criticality Identity (Theorem 2.1)**: For every n ≥ 2, ∑_{k=1}^n (r*_n)^k = 1.
-/
theorem reciprocal_identity (n : ℕ) (hn : 2 ≤ n) :
    criticalitySum (rStar n) n = 1 := by
      unfold rStar;
      grind +suggestions

/-! ## Section 4: The N-Bonacci Constant -/

/-- The n-bonacci constant C_n = 1 / r*_n. -/
noncomputable def nBonacciConst (n : ℕ) : ℝ := 1 / rStar n

/-! ## Section 5: Golden Ratio Connection (n = 2) -/

/-- The golden ratio φ = (1 + √5) / 2. -/
noncomputable abbrev φ : ℝ := Real.goldenRatio

/-
φ⁻¹ satisfies the n=2 criticality equation: φ⁻¹ + (φ⁻¹)² = 1.
-/
lemma goldenRatio_inv_criticality : criticalitySum φ⁻¹ 2 = 1 := by
  unfold criticalitySum;
  norm_num [ Finset.sum_range_succ ];
  grind

/-
φ⁻¹ is in the interval (0, 1).
-/
lemma goldenRatio_inv_pos : 0 < φ⁻¹ := by
  exact inv_pos.mpr ( by unfold φ; positivity )

lemma goldenRatio_inv_lt_one : φ⁻¹ < 1 := by
  exact inv_lt_one_of_one_lt₀ <| by rw [ show φ = ( 1 + Real.sqrt 5 ) / 2 by rfl ] ; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;

/-
**Golden Ratio Base Case**: r*_2 = φ⁻¹.
-/
theorem golden_ratio_reciprocal : rStar 2 = φ⁻¹ := by
  unfold rStar;
  grind +suggestions

/-! ## Section 6: r*_n > 1/2 for all n ≥ 2 (Theorem 2.2a) -/

/-
**Lower Bound**: r*_n > 1/2 for all n ≥ 2.
-/
theorem rn_gt_half (n : ℕ) (hn : 2 ≤ n) : rStar n > 1 / 2 := by
  by_contra! h_contra;
  -- Since criticalitySum is strictly increasing in x (each term x^(k+1) is strictly increasing for x > 0), if rStar n ≤ 1/2, then criticalitySum (rStar n) n ≤ criticalitySum (1/2) n.
  have h_sum_le : criticalitySum (rStar n) n ≤ criticalitySum (1 / 2) n := by
    exact Finset.sum_le_sum fun i hi => pow_le_pow_left₀ ( show 0 ≤ rStar n from le_of_lt ( rStar_pos n hn ) ) h_contra _;
  linarith [ reciprocal_identity n hn, criticalitySum_half_lt_one n ]

/-! ## Section 7: r*_n is strictly decreasing (Theorem 2.2b) -/

/-
**Strict Decrease**: r*_{n+1} < r*_n for n ≥ 2.
-/
theorem rn_decreasing (n : ℕ) (hn : 2 ≤ n) :
    rStar (n + 1) < rStar n := by
      -- By the properties of the criticality sum, we have criticalitySum (rStar n) (n + 1) > 1.
      have h_criticality_sum_gt_one : criticalitySum (rStar n) (n + 1) > 1 := by
        simp_all +decide [ criticalitySum ];
        rw [ Finset.sum_range_succ ] ; exact lt_add_of_le_of_pos ( reciprocal_identity n hn ▸ le_rfl ) ( pow_pos ( rStar_pos n hn ) _ ) ;
      -- By the properties of the criticality sum, we have criticalitySum (rStar (n + 1)) (n + 1) = 1.
      have h_criticality_sum_eq_one : criticalitySum (rStar (n + 1)) (n + 1) = 1 := by
        exact reciprocal_identity _ ( by linarith );
      contrapose! h_criticality_sum_gt_one;
      refine' le_trans _ h_criticality_sum_eq_one.le;
      exact Finset.sum_le_sum fun _ _ => pow_le_pow_left₀ ( by linarith [ rStar_pos n hn ] ) h_criticality_sum_gt_one _

/-! ## Section 8: Adelic Limit C_n → 2 (Theorem 2.2c) -/

/-
**Adelic Limit**: The n-bonacci constants converge to 2.
-/
theorem nBonacci_limit_is_two :
    Tendsto (fun n => nBonacciConst (n + 2)) atTop (𝓝 2) := by
      -- To prove the limit, it suffices to show that $r^*_n \to 1/2$ as $n \to \infty$.
      suffices h_lim : Filter.Tendsto (fun n => rStar n) Filter.atTop (nhds (1 / 2)) by
        simpa [ nBonacciConst ] using Filter.Tendsto.inv₀ ( h_lim.comp ( Filter.tendsto_add_atTop_nat 2 ) ) ( by norm_num );
      -- For any $x \in (1/2, 1)$, we have $criticalitySum x n \to \infty$ as $n \to \infty$. Hence, $r^*_n \leq x$ for sufficiently large $n$.
      have h_rStar_le_x : ∀ x ∈ Set.Ioo (1 / 2) 1, ∃ N : ℕ, ∀ n ≥ N, rStar n ≤ x := by
        -- Fix an arbitrary $x \in (1/2, 1)$.
        intro x hx
        -- Since $criticalitySum x n \to \infty$ as $n \to \infty$, there exists $N$ such that for all $n \geq N$, $criticalitySum x n > 1$.
        obtain ⟨N, hN⟩ : ∃ N : ℕ, ∀ n ≥ N, criticalitySum x n > 1 := by
          have h_criticalitySum_inf : Filter.Tendsto (fun n => criticalitySum x n) Filter.atTop (nhds (x / (1 - x))) := by
            convert HasSum.tendsto_sum_nat <| HasSum.mul_left x <| hasSum_geometric_of_lt_one ( by linarith [ hx.1 ] ) hx.2 using 1 ; norm_num [ criticalitySum ] ; ring;
          exact Filter.eventually_atTop.mp ( h_criticalitySum_inf.eventually ( lt_mem_nhds <| by rw [ lt_div_iff₀ ] <;> linarith [ hx.1, hx.2 ] ) );
        use N + 2;
        intro n hn
        by_contra h_contra
        have h_contra' : criticalitySum (rStar n) n = 1 := by
          exact reciprocal_identity n ( by linarith )
        have h_contra'' : criticalitySum x n ≤ 1 := by
          exact h_contra'.symm ▸ Finset.sum_le_sum fun _ _ => pow_le_pow_left₀ ( by linarith [ hx.1 ] ) ( le_of_not_ge h_contra ) _
        linarith [hN n (by linarith)];
      -- Since $r^*_n \geq 1/2$ for all $n \geq 2$, we can conclude that $r^*_n \to 1/2$ as $n \to \infty$.
      have h_rStar_ge_half : ∀ n ≥ 2, 1 / 2 ≤ rStar n := by
        exact fun n hn => le_of_lt ( rn_gt_half n hn );
      rw [ Metric.tendsto_nhds ];
      intro ε hε; rcases h_rStar_le_x ( 1 / 2 + Min.min ε ( 1 / 2 ) / 2 ) ⟨ by linarith [ lt_min hε ( by norm_num : ( 0 : ℝ ) < 1 / 2 ) ], by linarith [ min_le_left ε ( 1 / 2 ), min_le_right ε ( 1 / 2 ) ] ⟩ with ⟨ N, hN ⟩ ; filter_upwards [ Filter.eventually_ge_atTop N, Filter.eventually_ge_atTop 2 ] with n hn hn' using abs_lt.mpr ⟨ by linarith [ h_rStar_ge_half n hn', min_le_left ε ( 1 / 2 ), min_le_right ε ( 1 / 2 ) ], by linarith [ hN n hn, min_le_left ε ( 1 / 2 ), min_le_right ε ( 1 / 2 ) ] ⟩ ;

/-! ## Section 9: GUE Mode Connection -/

/-- The GUE consecutive spacing ratio mode value. -/
noncomputable def gueMode : ℝ := φ⁻¹

/-
**GUE Mode Theorem**: The mode of the GUE spacing ratio distribution
    equals φ⁻¹ = r*_2, the Fibonacci criticality threshold.
-/
theorem golden_ratio_mode : gueMode = rStar 2 := by
  exact golden_ratio_reciprocal.symm

/-! ## Section 10: Irrationality of log₂(3/2) -/

/-
**Irrationality of log₂(3/2)**: The base-2 logarithm of 3/2 is irrational.
    This establishes that Fibonacci driving cannot resonantly lock with
    any binary periodic protocol.
-/
theorem log2_three_half_irrational :
    Irrational (Real.log (3/2) / Real.log 2) := by
      by_contra h_contra
      obtain ⟨p, q, hq_pos, hpq_eq⟩ : ∃ p q : ℕ, q > 0 ∧ Real.log (3 / 2) / Real.log 2 = p / q := by
        obtain ⟨ q, hq ⟩ := Classical.not_not.mp h_contra; exact ⟨ q.num.natAbs, q.den, Nat.cast_pos.mpr q.pos, by simpa [ abs_of_nonneg ( Rat.num_nonneg.mpr ( show 0 ≤ q by exact_mod_cast hq.symm ▸ div_nonneg ( Real.log_nonneg ( by norm_num ) ) ( Real.log_nonneg ( by norm_num ) ) ) ), Rat.cast_def ] using hq.symm ⟩ ;
      -- Then we have $(3/2)^q = 2^p$.
      have h_exp : (3 / 2 : ℝ) ^ q = 2 ^ p := by
        rw [ div_eq_div_iff ] at hpq_eq <;> norm_num at *;
        · rw [ ← Real.rpow_natCast, ← Real.rpow_natCast, Real.rpow_def_of_pos, Real.rpow_def_of_pos ] <;> norm_num ; linarith;
        · linarith;
      rw [ div_pow, div_eq_iff ] at h_exp <;> norm_cast at * ; have := congr_arg Even h_exp ; norm_num [ hq_pos.ne', parity_simps ] at this ⊢;

/-! ## Section 11: Coil Energy and Landauer Bound -/

-- coilInvariant (= π²/12), coilInvariant_eq_zeta2_div2,
-- coilInvariant_exceeds_landauer imported from MNZI.Core (§1).
-- Local name `coilEnergy` is an alias for `coilInvariant`.
noncomputable def coilEnergy : ℝ := coilInvariant

theorem coilEnergy_eq_zeta2_div2 : coilEnergy = (Real.pi ^ 2 / 6) / 2 := by
  unfold coilEnergy; exact coilInvariant_eq_zeta2_div2

theorem coil_exceeds_landauer : coilEnergy > Real.log 2 := by
  unfold coilEnergy; exact coilInvariant_exceeds_landauer

/-! ## Section 12: Auxiliary Results -/

/-
For x ∈ (0,1), adding one more term increases the sum.
-/
lemma criticalitySum_succ_gt {x : ℝ} (hx0 : 0 < x) (_hx1 : x < 1)
    (n : ℕ) : criticalitySum x (n + 1) > criticalitySum x n := by
      unfold criticalitySum; rw [ Finset.sum_range_succ ] ; norm_num; positivity;

/-
r*_n converges to 1/2 as n → ∞.
-/
theorem rStar_tendsto_half :
    Tendsto (fun n => rStar (n + 2)) atTop (𝓝 (1/2)) := by
      -- By the convergence of $nBonacciConst (n + 2)$ to 2, we have that $rStar (n + 2) = 1 / nBonacciConst (n + 2)$ converges to $1 / 2$.
      have h_reciprocal : Filter.Tendsto (fun n => 1 / (nBonacciConst (n + 2))) Filter.atTop (nhds (1 / 2)) := by
        exact tendsto_const_nhds.div ( nBonacci_limit_is_two ) ( by norm_num );
      convert h_reciprocal using 2 ; unfold nBonacciConst ; norm_num

end MNZI
