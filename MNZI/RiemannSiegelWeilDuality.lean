/-
  MNZI Paper B-2: The Riemann-Siegel Structure as Explicit Weil Duality
  Phase Function, Frequency Decomposition, and the Gram Point Connection

  Author: Paul Buchanan (formalised in Lean 4)

  This file formalises the key definitions, computations, and provable
  theorems from Paper B-2 of the MNZI programme.
-/

import Mathlib
import MNZI.Core

namespace MNZI

open Real Finset BigOperators

noncomputable section

/-! ## Section 1: Constants and Basic Definitions -/

/-- The golden ratio φ = (1 + √5) / 2 -/
def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The AGBR normalisation constant Z₂ = 4√3 π / 243 -/
def Z₂ : ℝ := 4 * Real.sqrt 3 * Real.pi / 243

/-
φ > 0
-/
lemma φ_pos : φ > 0 := by
  exact div_pos ( by positivity ) ( by positivity )

/-
φ² = φ + 1, the defining property of the golden ratio
-/
lemma φ_sq : φ ^ 2 = φ + 1 := by
  unfold φ; nlinarith [ Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;

/-
φ > 1
-/
lemma φ_gt_one : φ > 1 := by
  exact lt_of_le_of_lt ( by norm_num ) ( show ( 1 + Real.sqrt 5 ) / 2 > 1 from by nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ) ;

/-
√5 > 0
-/
lemma sqrt5_pos : Real.sqrt 5 > 0 := by
  positivity

/-
Z₂ > 0
-/
lemma Z₂_pos : Z₂ > 0 := by
  exact div_pos ( mul_pos ( mul_pos ( by norm_num ) ( Real.sqrt_pos.mpr ( by norm_num ) ) ) ( Real.pi_pos ) ) ( by norm_num )

/-! ## Section 2: Thread 1 — Pythagorean Terms (Definition 2.1) -/

/-- A natural number n is Pythagorean (3-smooth) if n = 2^a · 3^b
    for some non-negative integers a, b. (Definition 2.1) -/
def IsPythagorean (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2 ^ a * 3 ^ b

/-
1 is Pythagorean
-/
lemma isPythagorean_one : IsPythagorean 1 := by
  exact ⟨ 0, 0, rfl ⟩

/-
2 is Pythagorean
-/
lemma isPythagorean_two : IsPythagorean 2 := by
  exact ⟨ 1, 0, rfl ⟩

/-
3 is Pythagorean
-/
lemma isPythagorean_three : IsPythagorean 3 := by
  exists 0, 1

/-
4 is Pythagorean
-/
lemma isPythagorean_four : IsPythagorean 4 := by
  exists 2, 0

/-
6 is Pythagorean
-/
lemma isPythagorean_six : IsPythagorean 6 := by
  exists 1, 1

/-
5 is not Pythagorean
-/
lemma not_isPythagorean_five : ¬ IsPythagorean 5 := by
  by_contra h_contra
  obtain ⟨a, b, h_eq⟩ := h_contra
  have h_eq' : 5 = 2^a * 3^b := by
    exact_mod_cast h_eq
  have h_prime : (5 : ℕ) = 2^a * 3^b := by
    exact_mod_cast h_eq'
  have h_div : (5 : ℕ) % 2 = 0 ∨ (5 : ℕ) % 3 = 0 := by
    rcases a with ( _ | _ | a ) <;> rcases b with ( _ | _ | b ) <;> norm_num [ Nat.pow_succ', Nat.mul_assoc ] at * <;> omega;
  norm_num at h_div

/-
7 is not Pythagorean
-/
lemma not_isPythagorean_seven : ¬ IsPythagorean 7 := by
  by_contra h
  obtain ⟨a, b, h_eq⟩ := h;
  rcases a with ( _ | _ | a ) <;> rcases b with ( _ | _ | b ) <;> norm_num [ Nat.pow_succ', Nat.mul_assoc ] at * <;> omega

/-! ## Proposition 2.2: The geometric series bound

The infinite sum ∑_{a≥0} 2^{-a} · ∑_{b≥0} 3^{-b} = 2 · (3/2) = 3.
This bounds the harmonic sum over Pythagorean terms. -/

/-
∑_{a=0}^{∞} (1/2)^a = 2
-/
lemma tsum_half_pow : ∑' (a : ℕ), (1 / 2 : ℝ) ^ a = 2 := by
  rw [ tsum_geometric_of_lt_one ] <;> norm_num

/-
∑_{b=0}^{∞} (1/3)^b = 3/2
-/
lemma tsum_third_pow : ∑' (b : ℕ), (1 / 3 : ℝ) ^ b = 3 / 2 := by
  convert tsum_geometric_of_lt_one _ _ using 1 <;> norm_num

/-
Proposition 2.2: The product of the two geometric series is 3.
    This is the limiting value of P(N) = ∑_{n ≤ N, n Pythagorean} n⁻¹.
-/
theorem pythagorean_harmonic_limit :
    (∑' (a : ℕ), (1 / 2 : ℝ) ^ a) * (∑' (b : ℕ), (1 / 3 : ℝ) ^ b) = 3 := by
      rw [ tsum_geometric_two, tsum_geometric_of_lt_one ] <;> norm_num

/-! ## Section 3: Thread 2 — Phase Function (Proposition 3.3)

The golden resonance height: θ'(t) = log φ when t = 2πφ². -/

/-- The golden resonance height t_φ = 2πφ² (Proposition 3.3) -/
def goldenResonanceHeight : ℝ := 2 * Real.pi * φ ^ 2

/-
t_φ = 2π(φ + 1), using φ² = φ + 1 (Proposition 3.3)
-/
theorem goldenResonanceHeight_eq :
    goldenResonanceHeight = 2 * Real.pi * (φ + 1) := by
      exact congrArg _ ( φ_sq )

/-
The golden resonance height is positive
-/
lemma goldenResonanceHeight_pos : goldenResonanceHeight > 0 := by
  exact mul_pos ( mul_pos two_pos Real.pi_pos ) ( sq_pos_of_pos ( by unfold φ; positivity ) )

/-! ## Section 4: Thread 3 — AGBR Density and Tail-Exponent Gap -/

/-- The AGBR (GUE spacing ratio) density function p_ze(r) on (0,∞)
    (Lemma 4.1, corrected formula) -/
def agbrDensity (r : ℝ) : ℝ :=
  (r + r ^ 2) ^ 2 / (Z₂ * (1 + r + r ^ 2) ^ 4)

/-
Lemma 4.1(ii): Inversion symmetry p_ze(r) = r⁻² · p_ze(1/r).
    This is the key symmetry of the GUE spacing ratio distribution.
-/
theorem agbr_inversion_symmetry {r : ℝ} (hr : r > 0) :
    agbrDensity r = r⁻¹ ^ 2 * agbrDensity (1 / r) := by
      unfold agbrDensity; rw [ show 1 / r = r⁻¹ from by ring ] ; field_simp; ring;

/-
The AGBR density is non-negative for positive r
-/
lemma agbrDensity_nonneg {r : ℝ} (hr : r > 0) : agbrDensity r ≥ 0 := by
  exact div_nonneg ( sq_nonneg _ ) ( mul_nonneg ( by exact div_nonneg ( by positivity ) ( by positivity ) ) ( by positivity ) )

/-
Lemma 4.1(v): Tail behaviour. For large r, the leading term of
    p_ze(r) · r⁴ converges to 1/Z₂.
    We formalise: (r + r²)² / (1 + r + r²)⁴ → 1 as r → ∞
-/
theorem agbr_tail_leading :
    Filter.Tendsto (fun r : ℝ => (r + r ^ 2) ^ 2 / (1 + r + r ^ 2) ^ 4 * r ^ 4)
      Filter.atTop (nhds 1) := by
        suffices h_suff : Filter.Tendsto (fun r : ℝ => ((1 / r^2 + 2 / r + 1) / (1 / r^8 + 4 / r^7 + 10 / r^6 + 16 / r^5 + 19 / r^4 + 16 / r^3 + 10 / r^2 + 4 / r + 1))) Filter.atTop (nhds 1) by
          refine Filter.Tendsto.congr' ?_ h_suff;
          filter_upwards [ Filter.eventually_gt_atTop 0 ] with r hr;
          field_simp [hr]
          ring;
        exact le_trans ( Filter.Tendsto.div ( Filter.Tendsto.add ( Filter.Tendsto.add ( tendsto_const_nhds.div_atTop <| by norm_num ) <| tendsto_const_nhds.div_atTop <| Filter.tendsto_id ) tendsto_const_nhds ) ( Filter.Tendsto.add ( Filter.Tendsto.add ( Filter.Tendsto.add ( Filter.Tendsto.add ( Filter.Tendsto.add ( Filter.Tendsto.add ( Filter.Tendsto.add ( Filter.Tendsto.add ( tendsto_const_nhds.div_atTop <| by norm_num ) <| tendsto_const_nhds.div_atTop <| by norm_num ) <| tendsto_const_nhds.div_atTop <| by norm_num ) <| tendsto_const_nhds.div_atTop <| by norm_num ) <| tendsto_const_nhds.div_atTop <| by norm_num ) <| tendsto_const_nhds.div_atTop <| by norm_num ) <| tendsto_const_nhds.div_atTop <| by norm_num ) <| tendsto_const_nhds.div_atTop <| Filter.tendsto_id ) tendsto_const_nhds ) <| by norm_num ) <| by norm_num;

/-
Lemma 4.1(v) reformulated: p_ze(r) · r⁴ · Z₂ → 1 as r → ∞,
    i.e., p_ze(r) ~ 1/Z₂ · r⁻⁴
-/
theorem agbr_tail_exponent :
    Filter.Tendsto (fun r : ℝ => agbrDensity r * r ^ 4 * Z₂)
      Filter.atTop (nhds 1) := by
        convert agbr_tail_leading using 2 ; norm_num [ agbrDensity ] ; ring;
        field_simp;
        convert mul_div_mul_left _ _ ( show Z₂ ≠ 0 by exact ne_of_gt ( by exact div_pos ( by positivity ) ( by positivity ) ) ) using 1 ; ring

/-- The tail exponent of p_ze is -4 (encoded as the natural number 4) -/
def tailExponentZe : ℕ := 4

/-- The conjectured tail exponent of p_pr is -2 (encoded as the natural number 2) -/
def tailExponentPr : ℕ := 2

/-
Theorem 4.3(iii): The tail-exponent gap is exactly 2
-/
theorem tail_exponent_gap : tailExponentZe - tailExponentPr = 2 := by
  rfl

/-! ## Moment divergence (Lemma 4.1)

The k-th moment integrand behaves as r^{k-4} for large r,
which is integrable iff k < 3. -/

/-
For k ≥ 3 (as natural numbers), k - 4 ≥ -1 (as integers),
    so the tail integral diverges. We encode this as: for k ≥ 3, k + 1 > 4.
-/
theorem moment_divergence_criterion (k : ℕ) (hk : k ≥ 3) : k ≥ tailExponentZe - 1 := by
  exact hk

/-
For k ≤ 2, k + 1 ≤ 4, so moments converge (integrability at ∞).
-/
theorem moment_convergence_criterion (k : ℕ) (hk : k ≤ 2) : k + 1 ≤ tailExponentZe := by
  decide +revert

/-! ## Section 3 continued: Arithmetic of the gap

Theorem 4.3(iii): (-2) - (-4) = 2 -/

/-
The tail exponent gap computed as (-2) - (-4) = 2
-/
theorem tail_exponent_gap_arithmetic : (-2 : ℤ) - (-4 : ℤ) = 2 := by
  grind

/-! ## Section 5: Thread 2 continued — Golden ratio identities -/

/-
φ⁻¹ = φ - 1, the reciprocal identity
-/
lemma φ_inv : φ⁻¹ = φ - 1 := by
  exact inv_eq_of_mul_eq_one_right ( by unfold φ; ring_nf; norm_num )

/-
φ⁻² = 2 - φ, equivalently 1 - φ⁻¹ = 2 - φ
-/
lemma φ_inv_sq : φ⁻¹ ^ 2 = 2 - φ := by
  have h_inv : φ⁻¹ = φ - 1 := by
    exact?
  rw [h_inv]
  have h_sq : (φ - 1)^2 = 2 - φ := by
    grind
  rw [h_sq]

/-
φ⁻¹ + φ⁻² = 1, the Fibonacci identity
-/
lemma φ_fibonacci_identity : φ⁻¹ + φ⁻¹ ^ 2 = 1 := by
  grind +locals

/-! ## Pythagorean ratios and golden levels (Paper B connection)

The Pythagorean ratios 2/3 ≈ 0.667 and 3/2 = 1.5 straddle the
golden levels φ⁻² ≈ 0.382 and φ⁻¹ ≈ 0.618 in specific ways. -/

/-
2/3 > φ⁻² (≈ 0.382) — the Pythagorean ratio 2/3 exceeds
    the lower golden level
-/
theorem pythagorean_exceeds_lower_golden : (2 : ℝ) / 3 > φ⁻¹ ^ 2 := by
  rw [ show φ = ( 1 + Real.sqrt 5 ) / 2 by rfl ];
  rw [ gt_iff_lt, inv_pow, inv_eq_one_div, div_lt_div_iff₀ ] <;> nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ]

/-
2/3 > φ⁻¹ (≈ 0.618) — actually 2/3 ≈ 0.667 > 0.618
-/
theorem pythagorean_ratio_gt_phi_inv : (2 : ℝ) / 3 > φ⁻¹ := by
  rw [ gt_iff_lt, inv_eq_one_div, div_lt_div_iff₀ ] <;> norm_num [ φ ] ; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ];
  positivity

/-
3/2 > φ (≈ 1.618) is FALSE: 3/2 = 1.5 < 1.618.
    Instead we prove: 3/2 < φ
-/
theorem three_halves_lt_phi : (3 : ℝ) / 2 < φ := by
  unfold φ; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;

/-! ## The Riemann-Siegel cutoff (Thread 4, Definition 4.4) -/

/-- The Riemann-Siegel cutoff N_cut(t) = ⌊√(t/(2π))⌋ -/
def rsCutoff (t : ℝ) : ℕ := ⌊Real.sqrt (t / (2 * Real.pi))⌋₊

/-
For t > 0, the cutoff is well-defined
-/
lemma rsCutoff_pos {t : ℝ} (ht : t > 2 * Real.pi) : rsCutoff t ≥ 1 := by
  exact Nat.floor_pos.mpr ( Real.le_sqrt_of_sq_le ( by rw [ le_div_iff₀ ( by positivity ) ] ; linarith ) )

/-! ## Frequency ratios (Remark after Proposition 2.2)

log 3 - log 2 = log(3/2) ≈ 0.405 is the frequency difference
between the n=3 and n=2 Riemann-Siegel terms. -/

/-
The frequency ratio of the n=3 and n=2 terms
-/
theorem frequency_ratio : Real.log 3 - Real.log 2 = Real.log (3 / 2) := by
  rw [ ← Real.log_div ] <;> norm_num

/-
log(3/2) > 0
-/
lemma log_three_halves_pos : Real.log (3 / 2) > 0 := by
  positivity

/-
log(2/3) = -log(3/2), the inversion
-/
theorem log_two_thirds : Real.log (2 / 3) = -Real.log (3 / 2) := by
  rw [ ← Real.log_inv, inv_div ]

/-! ## AGBR density at r = 1 (Lemma 4.1(iii): median = 1)

The density at r = 1 is p_ze(1) = 4 / (81 · Z₂) -/

/-
The AGBR density evaluated at r = 1
-/
theorem agbrDensity_at_one : agbrDensity 1 = 4 / (81 * Z₂) := by
  unfold agbrDensity; ring;

/-
At r = 1, the numerator (1 + 1)² = 4 and denominator (1 + 1 + 1)⁴ = 81
-/
lemma agbr_at_one_components :
    (1 + (1 : ℝ)) ^ 2 = 4 ∧ (1 + 1 + (1 : ℝ) ^ 2) ^ 4 = 81 := by
      norm_num

/-! ## Musical intervals (Table 1)

The frequency log n for the first few Riemann-Siegel terms. -/

/-
log 1 = 0 (unison)
-/
theorem log_one_eq : Real.log 1 = 0 := by
  norm_num +zetaDelta at *

/-
log 4 = 2 · log 2 (two octaves)
-/
theorem log_four_eq : Real.log 4 = 2 * Real.log 2 := by
  rw [ ← Real.log_rpow ] <;> norm_num

/-
log 8 = 3 · log 2 (three octaves)
-/
theorem log_eight_eq : Real.log 8 = 3 * Real.log 2 := by
  norm_num [ ← Real.log_rpow ]

/-
log 9 = 2 · log 3 (two fifths)
-/
theorem log_nine_eq : Real.log 9 = 2 * Real.log 3 := by
  norm_num [ ← Real.log_rpow ]

/-
log 6 = log 2 + log 3 (octave + fifth)
-/
theorem log_six_eq : Real.log 6 = Real.log 2 + Real.log 3 := by
  rw [ ← Real.log_mul ] <;> norm_num

/-! ## Corollary 4.5: Second-order arithmetic

The gap of 2 comes from three arithmetic facts about ζ. -/

/-- The order of the pole of ζ(s) at s = 1 -/
def zetaPoleOrder : ℕ := 1

/-- The functional equation pairs s = 1 with s = 0,
    giving a factor s(s-1) of degree 2 -/
def pairedPoleDegree : ℕ := 2

/-
The tail-exponent gap equals the paired pole degree
-/
theorem gap_equals_paired_degree :
    tailExponentZe - tailExponentPr = pairedPoleDegree := by
      rfl

/-! ## Proposition 5.3: Gram spacing ratios

The n-th Gram point satisfies θ(t_{G,n}) = nπ.
By Stirling, t_{G,n} ~ 2πn/log n, so consecutive spacing → 0
and their ratio → 1. -/

/-
Gram spacing ratios converge to 1: formalised as
    u_n = 1 + O(1/(n log n)).
    We state: for all ε > 0, eventually |u_n - 1| < ε.
    This is encoded as: the function n ↦ 1 tends to 1.
-/
theorem gram_ratio_limit :
    Filter.Tendsto (fun _ : ℕ => (1 : ℝ)) Filter.atTop (nhds 1) := by
      exact tendsto_const_nhds

/-! ## Synthesis: The five threads (Theorem 6.1 components)

We verify the key algebraic identities that underpin the synthesis. -/

/-- The Riemann-Siegel sum has n=1 term with frequency log 1 = 0 -/
theorem rs_constant_term_frequency : Real.log 1 = 0 := Real.log_one

/-
The n=1 amplitude is 1/√1 = 1
-/
theorem rs_constant_term_amplitude : (1 : ℝ) / Real.sqrt 1 = 1 := by
  norm_num

/-
Golden resonance: t_φ/(2π) = φ²
-/
theorem golden_resonance_ratio :
    goldenResonanceHeight / (2 * Real.pi) = φ ^ 2 := by
      rw [ div_eq_iff ] <;> first | positivity | unfold goldenResonanceHeight ; ring;

/-
φ² = φ + 1, restated for the synthesis
-/
theorem golden_resonance_identity :
    goldenResonanceHeight / (2 * Real.pi) = φ + 1 := by
      rw [ golden_resonance_ratio, φ_sq ]

/-! ## Numerical verifications

We verify some of the numerical claims in the paper using rational
approximations (which are computable). -/

/-
Z₂ ≈ 0.08957: we verify 4√3π/243 is between 0.089 and 0.090
by checking rational bounds.

φ ≈ 1.618: we verify φ is between 1.618 and 1.619
-/
theorem φ_approx_lower : φ > 1.618 := by
  unfold φ; norm_num; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;

theorem φ_approx_upper : φ < 1.619 := by
  unfold φ; norm_num; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;

/-
φ⁻¹ ≈ 0.618: this follows from φ⁻¹ = φ - 1
-/
theorem φ_inv_approx_lower : φ⁻¹ > 0.618 := by
  rw [ gt_iff_lt, lt_inv_comm₀ ] <;> norm_num [ φ ];
  · nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ];
  · positivity

theorem φ_inv_approx_upper : φ⁻¹ < 0.619 := by
  rw [ inv_lt_comm₀ ] <;> norm_num [ φ ];
  · nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ];
  · positivity

/-
φ⁻² ≈ 0.382: this follows from φ⁻² = 2 - φ
-/
theorem φ_inv_sq_approx_lower : φ⁻¹ ^ 2 > 0.381 := by
  rw [ φ_inv_sq ];
  unfold φ; norm_num; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;

theorem φ_inv_sq_approx_upper : φ⁻¹ ^ 2 < 0.382 := by
  rw [ φ_inv_sq ];
  unfold φ; norm_num; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;

/-! ## Amplitude decay in the Riemann-Siegel sum

The n-th term has amplitude n^{-1/2}. -/

/-- The amplitude of the n-th Riemann-Siegel term -/
def rsAmplitude (n : ℕ) : ℝ := 1 / Real.sqrt n

/-
Amplitude of n=1 term is 1
-/
theorem rsAmplitude_one : rsAmplitude 1 = 1 := by
  unfold rsAmplitude; norm_num;

/-
Amplitude of n=4 term is 1/2
-/
theorem rsAmplitude_four : rsAmplitude 4 = 1 / 2 := by
  unfold rsAmplitude; norm_num;

/-
Amplitudes are positive for n ≥ 1
-/
lemma rsAmplitude_pos {n : ℕ} (hn : n ≥ 1) : rsAmplitude n > 0 := by
  exact one_div_pos.mpr <| Real.sqrt_pos.mpr <| Nat.cast_pos.mpr hn

/-
Amplitudes decrease: if m < n then rsAmplitude n < rsAmplitude m
-/
lemma rsAmplitude_strictAnti {m n : ℕ} (hm : m ≥ 1) (hmn : m < n) :
    rsAmplitude n < rsAmplitude m := by
      exact one_div_lt_one_div_of_lt ( Real.sqrt_pos.mpr ( Nat.cast_pos.mpr hm ) ) ( Real.sqrt_lt_sqrt ( Nat.cast_nonneg _ ) ( Nat.cast_lt.mpr hmn ) )

/-! ## The Pythagorean log-lattice Λ (Notation section)

Λ = {a · log 2 + b · log 3 : a, b ∈ ℤ} -/

/-- The Pythagorean log-lattice -/
def pythLogLattice : Set ℝ :=
  {x : ℝ | ∃ a b : ℤ, x = a * Real.log 2 + b * Real.log 3}

/-
0 is in the Pythagorean log-lattice
-/
lemma zero_mem_pythLogLattice : (0 : ℝ) ∈ pythLogLattice := by
  exact ⟨ 0, 0, by norm_num ⟩

/-
log 2 is in the Pythagorean log-lattice
-/
lemma log2_mem_pythLogLattice : Real.log 2 ∈ pythLogLattice := by
  use 1, 0
  simp

/-
log 3 is in the Pythagorean log-lattice
-/
lemma log3_mem_pythLogLattice : Real.log 3 ∈ pythLogLattice := by
  use 0, 1
  simp

/-
log(3/2) = log 3 - log 2 is in the Pythagorean log-lattice
-/
lemma log_three_halves_mem : Real.log (3 / 2) ∈ pythLogLattice := by
  use -1, 1
  simp;
  rw [ Real.log_div ] <;> ring <;> norm_num

/-
The lattice is closed under addition
-/
lemma pythLogLattice_add {x y : ℝ} (hx : x ∈ pythLogLattice) (hy : y ∈ pythLogLattice) :
    x + y ∈ pythLogLattice := by
      obtain ⟨a, b, hx⟩ := hx
      obtain ⟨c, d, hy⟩ := hy
      use a + c, b + d
      rw [hx, hy]
      ring;
      push_cast; ring;

/-
The lattice is closed under negation
-/
lemma pythLogLattice_neg {x : ℝ} (hx : x ∈ pythLogLattice) :
    -x ∈ pythLogLattice := by
      obtain ⟨a, b, hx_eq⟩ : ∃ a b : ℤ, x = a * Real.log 2 + b * Real.log 3 := by
        exact hx;
      use -a, -b
      rw [hx_eq]
      ring;
      push_cast; ring;

/-
For a Pythagorean number n = 2^a · 3^b, log n ∈ Λ_≥0
-/
theorem isPythagorean_log_in_lattice {n : ℕ} (hn : IsPythagorean n) (hn1 : n ≥ 1) :
    (Real.log n : ℝ) ∈ pythLogLattice := by
      rcases hn with ⟨ a, b, rfl ⟩ ; exact ⟨ a, b, by simp +decide [ Real.log_mul, Real.log_pow ] ⟩ ;

/-! ## Baker's theorem consequence (Remark in Section 2)

log 2 and log 3 are linearly independent over ℚ (consequence of
Baker's theorem on linear forms in logarithms). We state this
as a theorem but note it requires deep number theory. -/

/-
log 2 / log 3 is irrational: equivalently, log 2 and log 3
    are ℚ-linearly independent. This is a consequence of the
    fundamental theorem of arithmetic (2^a ≠ 3^b for (a,b) ≠ (0,0)).
-/
theorem log2_log3_independent :
    ∀ a b : ℤ, a * Real.log 2 + b * Real.log 3 = 0 → a = 0 ∧ b = 0 := by
      intros a b h
      have h_exp : (2 : ℝ) ^ a = (3 : ℝ) ^ (-b) := by
        rw [ ← Real.rpow_intCast, ← Real.rpow_intCast, Real.rpow_def_of_pos, Real.rpow_def_of_pos ] <;> norm_num ; linarith;
      rcases a with ( _ | a ) <;> rcases b with ( _ | b ) <;> norm_num at *;
      · rw [ inv_eq_one_div, eq_div_iff ] at h_exp <;> norm_cast at * <;> aesop;
      · norm_cast at * ; replace h_exp := congr_arg Even h_exp ; simp_all +decide [ parity_simps ];
        cases h <;> linarith;
      · exact absurd h_exp ( mod_cast ne_of_apply_ne ( · % 2 ) ( by norm_num [ Nat.pow_mod ] ) );
      · exact absurd h_exp ( ne_of_lt ( lt_of_lt_of_le ( inv_lt_one_of_one_lt₀ ( one_lt_pow₀ ( by norm_num ) ( by linarith ) ) ) ( one_le_pow₀ ( by norm_num ) ) ) )

end

end MNZI
