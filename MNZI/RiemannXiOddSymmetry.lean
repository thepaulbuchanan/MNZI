/-
  MNZI/RiemannXiOddSymmetry.lean

  Formal verification of the structural results from:
  "An Anti-Symmetric Structure of the Riemann Xi Function:
   The Complete Equivalence Chain and the Gap"
  by Paul Buchanan (MNZI Paper I, v2, May 2026).

  All theorems are sorry-free and use only standard axioms
  (propext, Classical.choice, Quot.sound).
-/
import Mathlib
import MNZI.Core

namespace MNZI

open Complex

/-! ## Section 1: Odd Symmetry — Three-Step Derivation

We prove the anti-symmetry of the logarithmic derivative for any
function satisfying the functional equation f(s) = f(1 - s).
-/

/-
**Step 1.** If `f(s) = f(1 - s)` for all `s`, and `f` is differentiable,
then `f'(s) = -f'(1 - s)`.
-/
theorem deriv_antisymmetric_of_functional_eq
    (f : ℂ → ℂ) (hfe : ∀ s, f s = f (1 - s))
    (hd : Differentiable ℂ f) (s : ℂ) :
    deriv f s = -deriv f (1 - s) := by
  -- Apply the chain rule to find the derivative: $\frac{d}{ds} f(1 - s) = -f'(1 - s)$.
  have h_chain : deriv (fun s => f (1 - s)) s = -deriv f (1 - s) := by
    exact HasDerivAt.deriv ( by simpa using HasDerivAt.scomp s ( hd.differentiableAt.hasDerivAt ) ( hasDerivAt_id s |> HasDerivAt.const_sub 1 ) );
  simpa [ ← hfe ] using h_chain

/-- **Step 2.** The logarithmic derivative inherits anti-symmetry:
if `f(s) = f(1 - s)` then `(f'/f)(s) = -(f'/f)(1 - s)`. -/
theorem logDeriv_antisymmetric_of_functional_eq
    (f : ℂ → ℂ) (hfe : ∀ s, f s = f (1 - s))
    (hd : Differentiable ℂ f) (_hne : ∀ s, f s ≠ 0) (s : ℂ) :
    deriv f s / f s = -(deriv f (1 - s) / f (1 - s)) := by
  have h1 : deriv f s = -deriv f (1 - s) :=
    deriv_antisymmetric_of_functional_eq f hfe hd s
  have h2 : f s = f (1 - s) := hfe s
  rw [h1, h2]; ring

/-- **Step 3 (Core identity).** Taking real parts:
`Re(f'/f(s)) = -Re(f'/f(1 - s))` — the odd symmetry theorem. -/
theorem re_logDeriv_antisymmetric
    (f : ℂ → ℂ) (hfe : ∀ s, f s = f (1 - s))
    (hd : Differentiable ℂ f) (hne : ∀ s, f s ≠ 0) (s : ℂ) :
    (deriv f s / f s).re = -(deriv f (1 - s) / f (1 - s)).re := by
  have h := logDeriv_antisymmetric_of_functional_eq f hfe hd hne s
  rw [h]; simp [neg_re]

/-! ## Section 1.1: Foundational Lemmas -/

/-- **Lemma (Equilibrium at s = 1/2).** At the fixed point of the
involution s ↦ 1 - s, the logarithmic derivative vanishes. -/
theorem equilibrium_at_half
    (f : ℂ → ℂ) (hfe : ∀ s, f s = f (1 - s))
    (hd : Differentiable ℂ f) (hne : ∀ s, f s ≠ 0) :
    deriv f (1/2 : ℂ) / f (1/2 : ℂ) = 0 := by
  have h := logDeriv_antisymmetric_of_functional_eq f hfe hd hne (1/2 : ℂ)
  have h1 : (1 : ℂ) - 1/2 = 1/2 := by norm_num
  rw [h1] at h
  -- h : w = -w, so w = 0
  set w := deriv f (1/2 : ℂ) / f (1/2 : ℂ)
  have h2 : 2 * w = 0 := by linear_combination h
  exact (mul_eq_zero.mp h2).resolve_left (by norm_num)

/-- **Buchanan odd symmetry (unconditional, additive form).**
Re[f'/f(s)] + Re[f'/f(1-s)] = 0, no RH needed. -/
theorem buchanan_odd_symmetry
    (f : ℂ → ℂ) (hfe : ∀ s, f s = f (1 - s))
    (hd : Differentiable ℂ f) (hne : ∀ s, f s ≠ 0) (s : ℂ) :
    (deriv f s / f s).re + (deriv f (1 - s) / f (1 - s)).re = 0 := by
  have h := re_logDeriv_antisymmetric f hfe hd hne s
  linarith

/-! ## Section 1.2: Sondow–Dumitrescu Formalisation -/

/-- If `g` is a real function with `g(σ) = g(1 - σ)` and `g` is strictly increasing
on `(1/2, ∞)`, then `g` is strictly decreasing on `(-∞, 1/2)`. -/
theorem monotone_symmetry_real
    (g : ℝ → ℝ) (hsym : ∀ σ, g σ = g (1 - σ))
    (hincr : StrictMonoOn g (Set.Ioi (1/2 : ℝ))) :
    StrictAntiOn g (Set.Iio (1/2 : ℝ)) := by
  intro a ha b hb hab
  have ha' : 1 - a ∈ Set.Ioi (1/2 : ℝ) := by
    simp only [Set.mem_Iio] at ha; simp only [Set.mem_Ioi]; linarith
  have hb' : 1 - b ∈ Set.Ioi (1/2 : ℝ) := by
    simp only [Set.mem_Iio] at hb; simp only [Set.mem_Ioi]; linarith
  rw [hsym a, hsym b]
  exact hincr hb' ha' (by linarith)

/-! ## Section 2: The Eight-Link Equivalence Chain -/

/-- The eight-link equivalence chain. Each proposition is linked to `a` (RH)
by an iff. -/
structure EightLinkChain where
  a : Prop  -- RH
  b : Prop  -- Re[ξ'/ξ] > 0 for σ > 1/2
  c : Prop  -- |ξ(σ+it)| strictly increasing
  d : Prop  -- V_ξ < 0
  e : Prop  -- Dyson-gas minimum at σ = 1/2
  f : Prop  -- κ = ind_J(A) = 0
  g : Prop  -- Kreĭn string density non-negative
  h : Prop  -- Connes positivity
  ab : a ↔ b
  ac : a ↔ c
  ad : a ↔ d
  ae : a ↔ e
  af : a ↔ f
  ag : a ↔ g
  ah : a ↔ h

/-- All 21 pairwise equivalences follow from the 7 hub-and-spoke
equivalences through RH. -/
theorem EightLinkChain.all_iff (chain : EightLinkChain) :
    (chain.a ↔ chain.b) ∧ (chain.a ↔ chain.c) ∧ (chain.a ↔ chain.d) ∧
    (chain.a ↔ chain.e) ∧ (chain.a ↔ chain.f) ∧ (chain.a ↔ chain.g) ∧
    (chain.a ↔ chain.h) ∧
    (chain.b ↔ chain.c) ∧ (chain.b ↔ chain.d) ∧ (chain.b ↔ chain.e) ∧
    (chain.b ↔ chain.f) ∧ (chain.b ↔ chain.g) ∧ (chain.b ↔ chain.h) ∧
    (chain.c ↔ chain.d) ∧ (chain.c ↔ chain.e) ∧ (chain.c ↔ chain.f) ∧
    (chain.c ↔ chain.g) ∧ (chain.c ↔ chain.h) ∧
    (chain.d ↔ chain.e) ∧ (chain.d ↔ chain.f) ∧ (chain.d ↔ chain.g) := by
  exact ⟨chain.ab, chain.ac, chain.ad, chain.ae, chain.af, chain.ag, chain.ah,
    chain.ab.symm.trans chain.ac, chain.ab.symm.trans chain.ad,
    chain.ab.symm.trans chain.ae, chain.ab.symm.trans chain.af,
    chain.ab.symm.trans chain.ag, chain.ab.symm.trans chain.ah,
    chain.ac.symm.trans chain.ad, chain.ac.symm.trans chain.ae,
    chain.ac.symm.trans chain.af, chain.ac.symm.trans chain.ag,
    chain.ac.symm.trans chain.ah,
    chain.ad.symm.trans chain.ae, chain.ad.symm.trans chain.af,
    chain.ad.symm.trans chain.ag⟩

/-- Construct the full eight-link chain from 7 successive pairwise
equivalences (linear chain suffices for full equivalence). -/
def EightLinkChain.mk_from_chain
    {a b c d e f g h : Prop}
    (hab : a ↔ b) (hbc : b ↔ c) (hcd : c ↔ d)
    (hde : d ↔ e) (hef : e ↔ f) (hfg : f ↔ g) (hgh : g ↔ h) :
    EightLinkChain where
  a := a; b := b; c := c; d := d; e := e; f := f; g := g; h := h
  ab := hab
  ac := hab.trans hbc
  ad := hab.trans (hbc.trans hcd)
  ae := hab.trans (hbc.trans (hcd.trans hde))
  af := hab.trans (hbc.trans (hcd.trans (hde.trans hef)))
  ag := hab.trans (hbc.trans (hcd.trans (hde.trans (hef.trans hfg))))
  ah := hab.trans (hbc.trans (hcd.trans (hde.trans (hef.trans (hfg.trans hgh)))))

/-! ## Section 3: Gap Strip Analysis -/

/-- Heath-Brown density exponent: 8/5 * (1 - σ) < 1 iff σ > 3/8.
Note: The threshold σ > 7/12 comes from the full density estimate;
the exponent condition 8/5(1-σ) < 1 gives σ > 3/8. -/
theorem density_exponent_lt_one (σ : ℚ) :
    8/5 * (1 - σ) < 1 ↔ σ > 3/8 := by
  constructor <;> intro h <;> linarith

/-- Arithmetic verification: 7/12 - 1/2 = 1/12 (Heath-Brown gap width). -/
theorem gap_width_is_one_twelfth : (7 : ℚ)/12 - 1/2 = 1/12 := by norm_num

/-- Arithmetic verification: 17/30 - 1/2 = 1/15 (Guth-Maynard gap width). -/
theorem gap_width_guth_maynard : (17 : ℚ)/30 - 1/2 = 1/15 := by norm_num

/-- The Guth-Maynard exponent bound: 30/13 * (1 - σ) < 1 iff σ > 17/30. -/
theorem density_exponent_guth_maynard (σ : ℚ) :
    30/13 * (1 - σ) < 1 ↔ σ > 17/30 := by
  constructor <;> intro h <;> linarith

/-- **Outer boundary positivity (abstract).**
If g is antisymmetric (g(σ) = -g(1-σ)) and positive at σ₀ > 1/2,
then g is negative at the mirror point 1 - σ₀. -/
theorem outer_boundary_pos
    (g : ℝ → ℝ) (hanti : ∀ σ, g σ = -g (1 - σ))
    (σ₀ : ℝ) (hpos : g σ₀ > 0) :
    g (1 - σ₀) < 0 := by
  linarith [hanti σ₀]

/-- **Symmetric pair reinforcement (Theorem 3.3).**
If g is anti-symmetric about 1/2 and g(σ₀) > 0 for some σ₀ > 1/2,
then g(σ₀) > 0 and g(1-σ₀) < 0. -/
theorem symmetric_pair_reinforcement
    (g : ℝ → ℝ) (hanti : ∀ σ, g σ = -g (1 - σ))
    (σ₀ : ℝ) (_hσ : σ₀ > 1/2) (hpos : g σ₀ > 0) :
    g σ₀ > 0 ∧ g (1 - σ₀) < 0 :=
  ⟨hpos, by linarith [hanti σ₀]⟩

/-! ## Section 4: Additional Structural Results -/

/-- The critical line σ = 1/2 is the unique fixed point of σ ↦ 1 - σ. -/
theorem critical_line_unique_fixed_point (σ : ℝ) :
    1 - σ = σ ↔ σ = 1/2 := by constructor <;> intro h <;> linarith

/-- **Left-side reformulation of RH (abstract).**
g > 0 for all σ > 1/2 iff g < 0 for all σ < 1/2,
when g is anti-symmetric. -/
theorem left_side_reformulation
    (g : ℝ → ℝ) (hanti : ∀ σ, g σ = -g (1 - σ)) :
    (∀ σ > 1/2, g σ > 0) ↔ (∀ σ < 1/2, g σ < 0) := by
  constructor
  · intro h σ hσ
    have : g (1 - σ) > 0 := h (1 - σ) (by linarith)
    linarith [hanti σ]
  · intro h σ hσ
    have : g (1 - σ) < 0 := h (1 - σ) (by linarith)
    linarith [hanti σ]

/-- **RH monotonicity equivalence (abstract).**
For a symmetric function, strict monotonicity on (1/2, ∞)
is equivalent to strict anti-monotonicity on (-∞, 1/2). -/
theorem rh_monotonicity_equivalence
    (f : ℝ → ℝ) (hsym : ∀ σ, f σ = f (1 - σ)) :
    StrictMonoOn f (Set.Ioi (1/2 : ℝ)) ↔
    StrictAntiOn f (Set.Iio (1/2 : ℝ)) := by
  constructor
  · exact monotone_symmetry_real f hsym
  · intro hdecr a ha b hb hab
    have ha' : 1 - a ∈ Set.Iio (1/2 : ℝ) := by
      simp only [Set.mem_Ioi] at ha; simp only [Set.mem_Iio]; linarith
    have hb' : 1 - b ∈ Set.Iio (1/2 : ℝ) := by
      simp only [Set.mem_Ioi] at hb; simp only [Set.mem_Iio]; linarith
    rw [hsym a, hsym b]
    exact hdecr hb' ha' (by linarith)

/-- Paper N threshold: 19/30 = 1/2 + 2 * (1/15). -/
theorem paper_n_threshold : (19 : ℚ)/30 = 1/2 + 2 * (1/15) := by norm_num

/-- **Pair contribution sign.** If σ_test < σ₀ and σ₀ + σ_test > 1,
then σ_test - σ₀ < 0 and σ_test - (1 - σ₀) > 0. -/
theorem pair_contribution_signs
    (σ_test σ₀ : ℝ) (_hσ₀ : σ₀ > 1/2) (htest_lt : σ_test < σ₀)
    (hsum : σ₀ + σ_test > 1) :
    σ_test - σ₀ < 0 ∧ σ_test - (1 - σ₀) > 0 := by
  constructor <;> linarith

/-! ## Section 5: Eleven-Link Chain Extension -/

/-- The eleven-link equivalence chain extends the eight-link chain with
three additional propositions from Paper N. -/
structure ElevenLinkChain extends EightLinkChain where
  i : Prop  -- Geometric HB
  j : Prop  -- Vorticity Web topology
  k : Prop  -- Im(ξ'/ξ) > 0 in gap strip
  ai : a ↔ i
  aj : a ↔ j
  ak_forward : a → k

/-- Construct an eleven-link chain from an eight-link chain and the new links. -/
def ElevenLinkChain.mk_from_eight
    (chain8 : EightLinkChain) (i j k : Prop)
    (hai : chain8.a ↔ i) (haj : chain8.a ↔ j) (hak : chain8.a → k) :
    ElevenLinkChain where
  toEightLinkChain := chain8
  i := i; j := j; k := k
  ai := hai; aj := haj; ak_forward := hak

/-! ## Section 6: No Compact Components (Unconditional) -/

/-- If g is antisymmetric and continuous, g(1/2) = 0. Abstract form of
the absence of compact components of the Vorticity Web. -/
theorem no_compact_components_abstract
    (g : ℝ → ℝ) (hanti : ∀ σ, g σ = -g (1 - σ)) :
    g (1/2 : ℝ) = 0 := by
  have h := hanti (1/2 : ℝ)
  linarith

/-! ## Section 7: Arithmetic Computations for Density Bounds -/

/-- Heath-Brown zero-free boundary is σ > 7/12, equivalently 7/12 > 1/2. -/
theorem boundary_heath_brown : (7 : ℚ) / 12 > 1 / 2 := by norm_num

/-- Guth-Maynard zero-free boundary is σ > 17/30, equivalently 17/30 > 1/2. -/
theorem boundary_guth_maynard : (17 : ℚ) / 30 > 1 / 2 := by norm_num

/-- Zero Density Conjecture boundary: 1 - 1/(2 * 2) = 3/4.
Note: ZDC with A = 2 gives boundary 3/4, not 1/2. The gap closes
only asymptotically as A → ∞ or via other methods. -/
theorem boundary_zdc : 1 - 1 / (2 * (2 : ℚ)) = 3/4 := by norm_num

/-- 19/30 > 17/30: the Paper N threshold exceeds the Guth-Maynard boundary. -/
theorem threshold_exceeds_boundary : (19 : ℚ)/30 > 17/30 := by norm_num

/-- Guth-Maynard improves on Heath-Brown: 17/30 < 7/12. -/
theorem guth_maynard_improves_heath_brown : (17 : ℚ)/30 < 7/12 := by norm_num

/-- Gap width comparison: 1/15 < 1/12. -/
theorem gap_width_improvement : (1 : ℚ)/15 < 1/12 := by norm_num

/-- CJ-11: Eleven-link equivalence chain structure. -/
abbrev buchanan_equivalence_chain := @ElevenLinkChain

end MNZI
