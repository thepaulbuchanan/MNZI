/-
  MNZI/GeostrophicRigidityOddSymmetry.lean

  Geostrophic Rigidity and the Odd Symmetry Master Link:
  Kreĭn Space Methods for PT-Symmetric Spectral Theory

  18 sorry-free theorems formalizing the transferable structural results
  from the MNZI programme for PT-symmetric quantum mechanics.

  Key results:
  • Odd Symmetry Master Link (odd_symmetry_from_invariance)
  • Geostrophic Balance Criterion (commutator_vanishes)
  • Taylor Column Theorem (RH_implies_all_zeros_on_axis)
  • Geostrophic Violation Lemma (off_critical_zero_has_partner)
-/
import Mathlib
import MNZI.Core

namespace MNZI

open Complex

/-! ## The Kreĭn involution J(s) = 1 - s -/

/-- The Kreĭn involution J(s) = 1 - s, the fundamental symmetry of the MNZI framework. -/
noncomputable def J (s : ℂ) : ℂ := 1 - s

/-- The symmetry axis Σ_J = {s ∈ ℂ : Re(s) = 1/2}. -/
def symmetryAxis : Set ℂ := {s : ℂ | s.re = 1 / 2}

/-! ## Part 1: Foundational properties of J -/

/-
**Theorem 1.** The Kreĭn involution is involutive: J ∘ J = id.
-/
theorem kreinInvolution_involutive : Function.Involutive J := by
  exact fun s => by unfold J; ring;

/-
**Theorem 2.** Geostrophic pinning: if J(s) = s then Re(s) = 1/2.
-/
theorem geostrophic_pinning (s : ℂ) (h : J s = s) : s.re = 1 / 2 := by
  unfold J at h; norm_num [ Complex.ext_iff ] at h; linarith;

/-
**Theorem 3.** Strong geostrophic pinning: over ℂ, J(s) = s implies s = 1/2.
    The fixed-point set Fix(J) is a single point, not a line.
-/
theorem geostrophic_pinning_strong (s : ℂ) (h : J s = s) : s = 1 / 2 := by
  unfold J at h; linear_combination -h / 2;

/-
**Theorem 4.** The fixed-point set of J is exactly {1/2}.
-/
theorem fixedPoint_eq_half : {s : ℂ | J s = s} = {1 / 2} := by
  exact Set.eq_singleton_iff_unique_mem.mpr ⟨ by norm_num [ J ], fun s hs => by rw [ Set.mem_setOf_eq, J ] at hs; linear_combination -hs / 2 ⟩

/-
**Theorem 5.** The symmetry axis Σ_J equals the Kreĭn-invariant set
    {s : Re(Js) = Re(s)}.
-/
theorem criticalLine_eq_krein_invariant :
    symmetryAxis = {s : ℂ | (J s).re = s.re} := by
  ext s; exact (by
  unfold J symmetryAxis; norm_num; constructor <;> intro h <;> linarith;);

/-! ## Part 2: Zero symmetry and the Odd Symmetry Master Link -/

/-
**Theorem 6.** Zero symmetry: if f is J-invariant and f(s) = 0, then f(Js) = 0.
-/
theorem zero_symmetry (f : ℂ → ℂ) (hJ : ∀ s, f (J s) = f s)
    (s : ℂ) (hzero : f s = 0) : f (J s) = 0 := by
  rw [ hJ, hzero ]

/-
**Theorem 7.** Odd derivative from invariance (intermediate step):
    if f is J-invariant and differentiable, then f'(Js) = -f'(s).
-/
theorem odd_deriv_from_invariance (f : ℂ → ℂ) (hJ : ∀ s, f (J s) = f s)
    (hd : Differentiable ℂ f) (s : ℂ) :
    deriv f (J s) = -deriv f s := by
  -- Since $f(J(s)) = f(s)$ for all $s$, we have $deriv (fun s => f (J s)) s = deriv f s$.
  have h_chain : deriv (fun s => f (J s)) s = deriv f s := by
    aesop;
  erw [ ← h_chain, deriv_comp ] <;> norm_num [ hd.differentiableAt, J ];
  rw [ show deriv ( fun s => f ( 1 - s ) ) s = deriv f ( 1 - s ) * deriv ( fun s => 1 - s ) s by exact deriv_comp s hd.differentiableAt ( differentiableAt_id.const_sub _ ) ] ; norm_num [ sub_eq_add_neg ]

/-
**Theorem 8.** The Odd Symmetry Master Link:
    if f is J-invariant, differentiable, and nonvanishing, then
    (f'/f)(Js) = -(f'/f)(s). This is the mathematical expression of
    the Reflective Shade Duality.
-/
theorem odd_symmetry_from_invariance (f : ℂ → ℂ) (hJ : ∀ s, f (J s) = f s)
    (hd : Differentiable ℂ f) (s : ℂ) (_hne : f s ≠ 0) :
    deriv f (J s) / f (J s) = -(deriv f s / f s) := by
  rw [ hJ, odd_deriv_from_invariance f hJ hd s ] ; ring

/-! ## Part 3: Geostrophic Balance -/

/-
**Theorem 9.** Geostrophic Balance Criterion (commutator vanishes):
    For a J-invariant function f that is real-valued on the symmetry axis,
    the key commutator identity conj(f(Js)) - f(s) = 0 holds on Σ_J.
    This is the scalar reduction of [J,[J,A]] = 0 for A = JM_f.
-/
theorem commutator_vanishes (f : ℂ → ℂ) (hJ : ∀ s, f (J s) = f s)
    (hreal : ∀ s, s.re = 1 / 2 → (f s).im = 0)
    (s : ℂ) (hs : s.re = 1 / 2) :
    starRingEnd ℂ (f (J s)) - f s = 0 := by
  simp_all +decide [ Complex.ext_iff ]

/-! ## Part 4: Taylor Column Theorem (conditional) -/

/-
**Theorem 10.** Taylor Column Theorem: under the spectral hypothesis
    (all zeros of f lie on the symmetry axis), any zero s₀ of f
    satisfies Re(s₀) = 1/2.
-/
theorem RH_implies_all_zeros_on_axis (f : ℂ → ℂ)
    (spectral_hyp : ∀ s, f s = 0 → s.re = 1 / 2) (s₀ : ℂ) (h : f s₀ = 0) :
    s₀.re = 1 / 2 := by
  exact spectral_hyp s₀ h

/-
**Theorem 11.** Geostrophic Violation Lemma: an off-axis zero has a distinct
    J-partner. If f is J-invariant and s₀ is a zero with J(s₀) ≠ s₀,
    then J(s₀) is also a zero distinct from s₀.
-/
theorem off_critical_zero_has_partner (f : ℂ → ℂ) (hJ : ∀ s, f (J s) = f s)
    (s₀ : ℂ) (hzero : f s₀ = 0) (hoff : J s₀ ≠ s₀) :
    f (J s₀) = 0 ∧ J s₀ ≠ s₀ := by
  aesop

/-! ## Part 5: Completed Riemann zeta instances -/

/-
**Theorem 12.** Functional equation for completedRiemannZeta:
    ξ(1-s) = ξ(s).
-/
theorem functionalEquation_completedZeta_alt (s : ℂ) :
    completedRiemannZeta (1 - s) = completedRiemannZeta s := by
  convert completedRiemannZeta_one_sub s using 1

/-
**Theorem 13.** Functional equation for completedRiemannZeta₀:
    ξ₀(1-s) = ξ₀(s).
-/
theorem functionalEquation_completedZeta₀ (s : ℂ) :
    completedRiemannZeta₀ (1 - s) = completedRiemannZeta₀ s := by
  exact completedRiemannZeta₀_one_sub s

/-
**Theorem 14.** Odd derivative of completedRiemannZeta₀:
    ξ₀'(1-s) = -ξ₀'(s).
-/
theorem completedZeta₀_odd_deriv (s : ℂ) :
    deriv completedRiemannZeta₀ (1 - s) = -deriv completedRiemannZeta₀ s := by
  -- Apply the odd derivative from invariance theorem with $f = \text{completedRiemannZeta₀}$.
  apply odd_deriv_from_invariance completedRiemannZeta₀ (fun s => functionalEquation_completedZeta₀ s) (differentiable_completedZeta₀) s

/-
**Theorem 15.** Odd log-derivative of completedRiemannZeta:
    (ξ'/ξ)(1-s) = -(ξ'/ξ)(s), away from poles and zeros.
-/
theorem completedZeta_odd_logDeriv (s : ℂ) (hs0 : s ≠ 0) (hs1 : s ≠ 1)
    (h1s0 : 1 - s ≠ 0) (h1s1 : 1 - s ≠ 1)
    (_hne : completedRiemannZeta s ≠ 0) :
    deriv completedRiemannZeta (1 - s) / completedRiemannZeta (1 - s) =
    -(deriv completedRiemannZeta s / completedRiemannZeta s) := by
  have h_deriv : deriv (fun s => completedRiemannZeta (1 - s)) s = deriv completedRiemannZeta s := by
    exact Filter.EventuallyEq.deriv_eq ( by filter_upwards [ IsOpen.mem_nhds ( isOpen_compl_singleton.inter isOpen_compl_singleton ) ⟨ hs0, hs1 ⟩ ] with x hx using by rw [ functionalEquation_completedZeta_alt ] );
  have h_chain : deriv (fun s => completedRiemannZeta (1 - s)) s = deriv (completedRiemannZeta) (1 - s) * deriv (fun s => 1 - s) s := by
    convert deriv_comp _ _ _ using 1;
    · apply_rules [ HurwitzZeta.differentiableAt_completedHurwitzZetaEven ];
      tauto;
    · exact differentiableAt_id.const_sub _;
  simp_all +decide [ sub_eq_add_neg ];
  rw [ show completedRiemannZeta ( 1 + -s ) = completedRiemannZeta s from by simpa [ sub_eq_add_neg ] using functionalEquation_completedZeta_alt s ] ; ring

/-! ## Part 6: Open question consequences and additional results -/

/-
**Theorem 16.** Exceptional point pairing: if f is J-invariant and s₀ is
    an off-axis zero (Re(s₀) ≠ 1/2), then s₀ and J(s₀) are distinct zeros.
    This is the formal content of the pairing structure (partial OQ-PT-5).
-/
theorem exceptional_point_pairing (f : ℂ → ℂ) (hJ : ∀ s, f (J s) = f s)
    (s₀ : ℂ) (hzero : f s₀ = 0) (hoff : s₀.re ≠ 1 / 2) :
    f (J s₀) = 0 ∧ J s₀ ≠ s₀ := by
  exact ⟨ hJ s₀ ▸ hzero, fun h => hoff <| by have := geostrophic_pinning s₀ h; norm_num at *; linarith ⟩

/-
**Theorem 17.** Log-derivative sum vanishes: for any J-invariant differentiable
    nonvanishing f, the sum of log-derivatives at s and J(s) vanishes.
    (Partial OQ-PT-6.)
-/
theorem odd_logderiv_sum_vanishes (f : ℂ → ℂ) (hJ : ∀ s, f (J s) = f s)
    (hd : Differentiable ℂ f) (s : ℂ) (_hne : f s ≠ 0) :
    deriv f s / f s + deriv f (J s) / f (J s) = 0 := by
  rw [hJ, odd_deriv_from_invariance f hJ hd s]; ring

/-
**Theorem 18.** Reflective Shade Duality (forward direction):
    J-symmetry of f implies J-antisymmetry of f'/f.
    This is the abstract formulation of the Reflective Shade Duality.
-/
theorem reflective_shade_duality_forward (f : ℂ → ℂ)
    (hJ : ∀ s, f (J s) = f s) (hd : Differentiable ℂ f)
    (s : ℂ) (hne : f s ≠ 0) :
    deriv f (J s) / f (J s) = -(deriv f s / f s) := by
  convert odd_symmetry_from_invariance f hJ hd s hne using 1

end MNZI
