import Mathlib
import MNZI.Core

/-!
# Golden Ratio and Prime Gap Ratios: The Straddle Theorem

We prove that the empirical CDF of consecutive prime gap ratios
  r(n) = (p(n+2) - p(n+1)) / (p(n+1) - p(n))
straddles φ⁻² = (3 - √5)/2 = 2 - φ at the threshold 2/3,
for a sample of N = 1,270,605 consecutive primes.

Specifically:
  #{n ≤ N : r(n) < 2/3} / N  <  φ⁻²  <  #{n ≤ N : r(n) ≤ 2/3} / N

The proof has two components:
1. **Computational**: Exact integer arithmetic via a prime sieve verifies the counts
   (456,900 strict-below and 500,886 at-or-below) using `native_decide`.
2. **Algebraic**: The comparison with the irrational φ⁻² = (3 - √5)/2 is certified
   by squaring to eliminate √5, reducing to decidable integer arithmetic.
-/

open Real

namespace MNZI.GoldenPrime

/-! ## Computational Component -/

/-- Sieve of Eratosthenes. Returns `ByteArray` of size `bound + 1`
    where entry `i` is `1` iff `i` is prime. -/
def mkSieve (bound : Nat) : ByteArray := Id.run do
  let mut s := ByteArray.mk (Array.replicate (bound + 1) (1 : UInt8))
  s := s.set! 0 0
  if bound ≥ 1 then s := s.set! 1 0
  for i in List.range (bound + 1) do
    if i < 2 then continue
    if i * i > bound then break
    if s.get! i == (1 : UInt8) then
      let cnt := (bound - i * i) / i + 1
      for k in List.range cnt do
        s := s.set! (i * i + k * i) 0
  return s

/-- Collect the first `count` primes using a sieve up to `bound`. -/
def collectPrimes (count bound : Nat) : Array Nat := Id.run do
  let s := mkSieve bound
  let mut ps : Array Nat := Array.mkEmpty count
  for i in List.range (bound + 1) do
    if ps.size ≥ count then break
    if s.get! i == (1 : UInt8) then
      ps := ps.push i
  return ps

/-- The sample size N = 1,270,605. -/
abbrev N : Nat := 1270605

/-- Compute `(countBelow, countAtOrBelow)` for the gap ratio threshold `num/den`.
    - `countBelow`     = #{1 ≤ n ≤ N : den·(p(n+2)-p(n+1)) < num·(p(n+1)-p(n))}
    - `countAtOrBelow` = #{1 ≤ n ≤ N : den·(p(n+2)-p(n+1)) ≤ num·(p(n+1)-p(n))} -/
def gapCounts (num den : Nat) : Nat × Nat := Id.run do
  let ps := collectPrimes (N + 2) 20000100
  let mut cB : Nat := 0
  let mut cAB : Nat := 0
  for k in List.range N do
    let g1 := ps[k + 1]! - ps[k]!
    let g2 := ps[k + 2]! - ps[k + 1]!
    if den * g2 < num * g1 then
      cB := cB + 1
      cAB := cAB + 1
    else if den * g2 == num * g1 then
      cAB := cAB + 1
  return (cB, cAB)

/-- The exact gap ratio counts at threshold 2/3, verified by native computation.
    Uses `Lean.ofReduceBool` (the axiom behind `native_decide`). -/
theorem gapCounts_val : gapCounts 2 3 = (456900, 500886) := by native_decide

/-! ## CDF Definitions -/

/-- Left limit of empirical CDF at `q`: `F_N(q⁻) = #{n ≤ N : r(n) < q} / N`. -/
noncomputable def F_N (q : ℚ) (n : Nat := N) : ℝ :=
  ((gapCounts q.num.natAbs q.den).1 : ℝ) / n

/-- Empirical CDF value at `q`: `F_N(q) = #{n ≤ N : r(n) ≤ q} / N`. -/
noncomputable def F_N_right (q : ℚ) (n : Nat := N) : ℝ :=
  ((gapCounts q.num.natAbs q.den).2 : ℝ) / n

/-! ## Algebraic Component -/

/-
Upper bound on √5 scaled: `1270605 · √5 < 2898015`.

    Proof by squaring: `1270605² · 5 = 8072185330125 < 8398490940225 = 2898015²`.
-/
lemma sqrt5_mul_upper : (1270605 : ℝ) * Real.sqrt 5 < 2898015 := by
  nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ]

/-
Lower bound on √5 scaled: `2810043 < 1270605 · √5`.

    Proof by squaring: `2810043² = 7896341661849 < 8072185330125 = 1270605² · 5`.
-/
lemma sqrt5_mul_lower : (2810043 : ℝ) < 1270605 * Real.sqrt 5 := by
  nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ]

/-! ## Main Theorem -/

open goldenRatio in
/-- The empirical CDF of consecutive prime gap ratios straddles
    `φ⁻² = 2 - φ = (3 - √5)/2` at threshold `2/3` for `N = 1,270,605`.

    This means the golden-ratio squared inverse lies strictly between
    the left-limit and the value of the empirical CDF at `2/3`. -/
theorem straddle_two_thirds :
    F_N (2 / 3 : ℚ) < (2 - φ : ℝ) ∧ (2 - φ : ℝ) < F_N_right (2 / 3 : ℚ) := by
  have h_gapCounts : gapCounts 2 3 = (456900, 500886) := by
    exact gapCounts_val
  have h_ineq1 : 456900 / 1270605 < 2 - goldenRatio := by
    unfold goldenRatio; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;
  have h_ineq2 : 2 - goldenRatio < 500886 / 1270605 := by
    unfold goldenRatio; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;
  unfold F_N F_N_right; norm_num [ h_gapCounts, h_ineq1, h_ineq2 ] ;
  constructor <;> linarith

/-- CJ-29: Buchanan Golden Straddle (Certified).
    The straddle F(2/3⁻) < φ⁻² < F(2/3) formally certified
    by native computation over N = 1,270,605 consecutive prime
    gap ratios below 2 × 10⁷.

    Proof architecture:
    - Computational: gapCounts_val verified by native_decide
      (exact integer arithmetic, no floating point)
    - Algebraic: integer witnesses a = 2,898,015 and b = 2,810,043
      satisfy a² = 8,398,490,940,225 > 5N² = 8,072,185,330,125
      and b² = 7,896,341,661,849 < 5N² respectively,
      certified by norm_num via Real.sq_sqrt and nlinarith.

    Axioms: propext, Classical.choice, Lean.ofReduceBool,
    Lean.trustCompiler, Quot.sound.
    Note: Lean.ofReduceBool is the standard axiom underlying
    native_decide; it trusts compiled native evaluation for
    computationally intensive decidable propositions.

    Source: MNZI/GoldenPrime.lean (Paper B, Session 40).
    Independent verification: straddle_certificate.py (SHA-256:
    45524a64236a2d6173c2604e247bbae4c2dc7bd91866ce0972b2d40adcd47dba) -/
def buchanan_golden_straddle_certified := @straddle_two_thirds

end MNZI.GoldenPrime
