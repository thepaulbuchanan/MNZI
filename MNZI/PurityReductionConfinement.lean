import Mathlib

/-!
# Purity, Reduction, and Confinement in the Riemann Derivative Tower

Lean 4 formalization of the main algebraic results from the paper
"Purity, Reduction, and Confinement in the Riemann Derivative Tower"
by Paul Buchanan (MNZI Programme, June 2026).

Since the Riemann ξ function's full derivative API is not available in Mathlib,
we prove abstract algebraic versions of the theorems. These apply to any function
satisfying the functional equation symmetry ξ(s) = ξ(1−s) and the Schwarz
reflection condition conj(ξ(s)) = ξ(conj(s)), via their consequence on the
critical line: ξ^(k)(1/2+it) = (−1)^k · conj(ξ^(k)(1/2+it)).

## Main Results

* **Alternating Parity Theorem** (`purity_alternating_parity_real`, `purity_alternating_parity_im`):
  If z = (−1)^k · conj(z), then z is real for even k and purely imaginary for odd k.

* **Critical Line Purity Lemma** (`purity_real_div_im`, `purity_im_div_real`):
  The ratio of a real and a purely imaginary number is purely imaginary.

* **Conjugation Lemma** (`rk_conjugation_lemma`):
  R_k(1+it) = −conj(R_k(it)), derived algebraically from the functional equation.

* **Unit Modulus** (`confinement_unit_modulus`):
  |z/(−conj z)| = 1, giving |φ^(k)| = 1 as a three-line consequence.

* **Confinement Characterization** (`confinement_neg_one_iff`, `confinement_plus_i_iff`):
  Algebraic conditions for φ^(k) ∈ {+i, −1}.

* **Fixed/Anti-fixed Point Geometry** (`confinement_fixed_points`, `confinement_antifixed_points`):
  Classification of fixed and anti-fixed points of conjugation on S¹.

## References

All results correspond to Sections 2–4 of the paper. The abstract approach
(proving for arbitrary z satisfying the parity condition, then instantiating)
follows the suggestion in Open Question OQ-R2-7 of the paper.
-/

namespace MNZI

open Complex

/-- Notation for complex conjugation -/
local notation "conj" => starRingEnd ℂ

/-! ## Section 1: Complex Conjugation Algebra -/

theorem purity_conj_eq_self_iff (z : ℂ) : conj z = z ↔ z.im = 0 := by
  simp +decide [ Complex.ext_iff, Complex.conj_ofReal ];
  constructor <;> intro h <;> linarith

theorem purity_conj_eq_neg_iff (z : ℂ) : conj z = -z ↔ z.re = 0 := by
  simp +decide [ Complex.ext_iff ];
  grind

/-! ## Section 2: Alternating Parity Theorem -/

theorem purity_neg_one_pow_even (k : ℕ) (hk : Even k) : (-1 : ℂ) ^ k = 1 := by
  rw [ hk.neg_one_pow ]

theorem purity_neg_one_pow_odd (k : ℕ) (hk : Odd k) : (-1 : ℂ) ^ k = -1 := by
  exact hk.neg_one_pow

theorem purity_alternating_parity_real (z : ℂ) (k : ℕ) (hk : Even k)
    (h : z = (-1 : ℂ) ^ k * conj z) : z.im = 0 := by
  rw [ eq_comm ] at h;
  rw [ ← purity_conj_eq_self_iff ] ; simp_all +decide [ hk.neg_one_pow ]

theorem purity_alternating_parity_im (z : ℂ) (k : ℕ) (hk : Odd k)
    (h : z = (-1 : ℂ) ^ k * conj z) : z.re = 0 := by
  norm_num [ hk.neg_one_pow ] at h; exact purity_conj_eq_neg_iff z |>.1 <| by linear_combination h;

theorem purity_alternating_parity (z : ℂ) (k : ℕ)
    (h : z = (-1 : ℂ) ^ k * conj z) :
    (Even k → z.im = 0) ∧ (Odd k → z.re = 0) :=
  ⟨fun hk => purity_alternating_parity_real z k hk h,
   fun hk => purity_alternating_parity_im z k hk h⟩

/-! ## Section 3: Critical Line Purity Lemma -/

theorem purity_real_div_im (a b : ℂ) (ha : a.im = 0) (hb : b.re = 0)
    (hb' : b ≠ 0) : (a / b).re = 0 := by
  simp_all +decide [ Complex.div_re ]

theorem purity_im_div_real (a b : ℂ) (ha : a.re = 0) (hb : b.im = 0)
    (hb' : b ≠ 0) : (a / b).re = 0 := by
  simp_all +decide [ Complex.div_re ]

theorem purity_critical_line (num den : ℂ) (k : ℕ) (hden : den ≠ 0)
    (hnum : num = (-1 : ℂ) ^ (k + 1) * conj num)
    (hden_eq : den = (-1 : ℂ) ^ k * conj den) :
    (num / den).re = 0 := by
  rcases Nat.even_or_odd' k with ⟨ k, rfl | rfl ⟩ <;> norm_num [ pow_add ] at *;
  · norm_num [ Complex.ext_iff ] at *;
    norm_num [ Complex.div_re, show num.re = 0 by linarith, show den.im = 0 by linarith ];
  · norm_num [ Complex.ext_iff ] at *;
    norm_num [ Complex.div_re, show num.im = 0 by linarith, show den.re = 0 by linarith ]

/-! ## Section 4: Conjugation Lemma -/

theorem rk_conjugation_lemma (fk1 fk gk1 gk : ℂ) (k : ℕ)
    (hfk1 : fk1 = (-1 : ℂ) ^ (k + 1) * conj gk1)
    (hfk : fk = (-1 : ℂ) ^ k * conj gk)
    (hgk : gk ≠ 0) :
    fk1 / fk = -(conj (gk1 / gk)) := by
  simp +decide [ *, pow_succ', mul_assoc, mul_comm, mul_left_comm, div_eq_mul_inv ]

/-! ## Section 5: Unit Modulus from Conjugation -/

theorem confinement_unit_modulus (z : ℂ) (hz : z ≠ 0) :
    ‖z / (-(conj z))‖ = 1 := by
  norm_num [ hz ]

theorem confinement_on_unit_circle (z : ℂ) (hz : z ≠ 0) :
    normSq (z / (-(conj z))) = 1 := by
  simp +decide [ hz, Complex.normSq_eq_norm_sq ]

/-! ## Section 6: Confinement Characterization -/

theorem confinement_quot_eq_one_iff (z : ℂ) (hz : z ≠ 0) :
    z / (-(conj z)) = 1 ↔ z.re = 0 := by
  rw [ div_eq_iff ] <;> norm_num [ Complex.ext_iff ];
  · grind +extAll;
  · exact fun h₁ h₂ => hz <| Complex.ext h₁ h₂

theorem confinement_neg_one_iff (z : ℂ) (hz : z ≠ 0) :
    z / (-(conj z)) = -1 ↔ z.im = 0 := by
  rw [ div_eq_iff_mul_eq ];
  · simp +decide [ Complex.ext_iff ];
    constructor <;> intro h <;> linarith;
  · aesop

theorem confinement_plus_i_iff (z : ℂ) (hz : z ≠ 0) :
    z / (-(conj z)) = I ↔ z.re + z.im = 0 := by
  norm_num [ Complex.ext_iff, div_eq_iff, hz ];
  grind +locals

theorem confinement_neg_i_iff (z : ℂ) (hz : z ≠ 0) :
    z / (-(conj z)) = -I ↔ z.re = z.im := by
  rw [ div_eq_iff ] <;> norm_num [ Complex.ext_iff ] at * ; aesop;
  exact hz

theorem confinement_exact (z : ℂ) (hz : z ≠ 0) :
    z / (-(conj z)) = I ∨ z / (-(conj z)) = -1 ↔
    z.re + z.im = 0 ∨ z.im = 0 := by
  constructor
  · rintro (h | h)
    · left; exact (confinement_plus_i_iff z hz).mp h
    · right; exact (confinement_neg_one_iff z hz).mp h
  · rintro (h | h)
    · left; exact (confinement_plus_i_iff z hz).mpr h
    · right; exact (confinement_neg_one_iff z hz).mpr h

theorem confinement_neg_i_exclusion (z : ℂ) (hz : z ≠ 0) :
    z / (-(conj z)) ≠ -I ↔ z.re ≠ z.im :=
  (confinement_neg_i_iff z hz).not

/-! ## Section 7: Fixed and Anti-Fixed Point Geometry of S¹ -/

theorem confinement_antifixed_i : conj I = -I := by
  norm_num [ Complex.ext_iff ]

theorem confinement_fixed_neg_one : conj (-1 : ℂ) = -1 := by
  norm_num [ Complex.ext_iff, Complex.conj_ofReal ]

theorem confinement_i_on_circle : ‖(I : ℂ)‖ = 1 := by
  norm_num

theorem confinement_neg_one_on_circle : ‖(-1 : ℂ)‖ = 1 := by
  norm_num

theorem confinement_fixed_points (z : ℂ) (hz : ‖z‖ = 1) :
    conj z = z ↔ z = 1 ∨ z = -1 := by
  simp_all +decide [ Complex.ext_iff ];
  norm_num [ Complex.normSq, Complex.norm_def ] at hz;
  exact ⟨ fun h => by cases le_or_gt 0 z.re <;> [ left; right ] <;> constructor <;> nlinarith, fun h => by cases h <;> nlinarith ⟩

theorem confinement_antifixed_points (z : ℂ) (hz : ‖z‖ = 1) :
    conj z = -z ↔ z = I ∨ z = -I := by
  norm_num [ Complex.norm_def, Complex.normSq_apply, Complex.ext_iff ] at *;
  grind

theorem confinement_set_structure :
    (conj I = -I ∧ conj (-1 : ℂ) = (-1 : ℂ)) := by
  exact ⟨confinement_antifixed_i, confinement_fixed_neg_one⟩

/-! ## Section 8: The Doubling Map Identity -/

theorem confinement_doubling_identity (z : ℂ) (hz : z ≠ 0) :
    z / (-(conj z)) = -(z ^ 2 / (normSq z : ℂ)) := by
  simp +decide [ div_eq_mul_inv, sq, Complex.ext_iff, hz ];
  constructor <;> ring

theorem confinement_quot_sq (z : ℂ) (hz : z ≠ 0) :
    (z / (-(conj z))) ^ 2 = (z / conj z) ^ 2 := by
  ring

/-! ## Section 9: Abstract Functional Equation Framework -/

structure AbstractSymmetricDeriv where
  val : ℂ
  order : ℕ
  parity : val = (-1 : ℂ) ^ order * conj val

theorem purity_abstract_even_real (f : AbstractSymmetricDeriv) (hk : Even f.order) :
    f.val.im = 0 :=
  purity_alternating_parity_real f.val f.order hk f.parity

theorem purity_abstract_odd_im (f : AbstractSymmetricDeriv) (hk : Odd f.order) :
    f.val.re = 0 :=
  purity_alternating_parity_im f.val f.order hk f.parity

theorem purity_abstract_base_case (f : AbstractSymmetricDeriv) (hk : f.order = 0) :
    f.val.im = 0 :=
  purity_abstract_even_real f ⟨0, by omega⟩

theorem purity_abstract_purity (f g : AbstractSymmetricDeriv)
    (hfg : f.order = g.order + 1) (hg : g.val ≠ 0) :
    (f.val / g.val).re = 0 :=
  purity_critical_line f.val g.val g.order hg (hfg ▸ f.parity) g.parity

end MNZI
