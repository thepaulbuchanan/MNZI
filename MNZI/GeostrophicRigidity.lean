/-
  MNZI/GeostrophicRigidity.lean

  Geostrophic Rigidity of the Eisenstein Scattering Family:
  A Taylor–Proudman Theorem for the Riemann Zeros (Paper O)

  Machine-verified formalization of the fluid-dynamic / spectral-theory
  dictionary connecting the Taylor–Proudman theorem to the functional
  equation of the completed Riemann zeta function.
-/
import Mathlib
import MNZI.Core

open Complex

namespace MNZI

/-! ## Section 1: Kreĭn Involution -/

/-- The Kreĭn involution J : ℂ → ℂ defined by J(s) = 1 - s.
    Analogue of the Coriolis reflection in the fluid-dynamic dictionary. -/
noncomputable def kreinInvolution (s : ℂ) : ℂ := 1 - s

/-- kreinInvolution agrees with functionalEqInvolution from Core. -/
theorem kreinInvolution_eq_functionalEqInvolution (s : ℂ) :
    kreinInvolution s = functionalEqInvolution s := by
  unfold kreinInvolution functionalEqInvolution; rfl

/-
J is an involution: J(J(s)) = s.
-/
theorem kreinInvolution_involutive : Function.Involutive kreinInvolution := by
  exact fun s => by unfold kreinInvolution; ring

/-
J is bijective (immediate from involutivity).
-/
theorem kreinInvolution_bijective : Function.Bijective kreinInvolution := by
  exact Function.Involutive.bijective kreinInvolution_involutive

/-! ## Section 2: Critical Line -/

-- OnCriticalLine is imported from MNZI.Core:
--   def OnCriticalLine (s : ℂ) : Prop := s.re = 1 / 2

/-
J preserves the critical line: if Re(s) = 1/2 then Re(1-s) = 1/2.
-/
theorem kreinInvolution_preserves_criticalLine (s : ℂ) (h : OnCriticalLine s) :
    OnCriticalLine (kreinInvolution s) := by
  unfold OnCriticalLine at *; unfold kreinInvolution; norm_num [Complex.ext_iff] at *; linarith

/-
The critical line is the unique J-invariant axis:
    Re(J(s)) = Re(s) ↔ Re(s) = 1/2.
-/
theorem criticalLine_eq_krein_invariant (s : ℂ) :
    (kreinInvolution s).re = s.re ↔ OnCriticalLine s := by
  unfold kreinInvolution OnCriticalLine; norm_num; constructor <;> intro <;> linarith

/-! ## Section 3: Geostrophic Fluid Dictionary -/

/-- Abstract structure packaging the fluid-dynamic / spectral-theory dictionary. -/
structure GeostrophicFluidDictionary where
  involution : ℂ → ℂ
  scatteringCoeff : ℂ → ℂ
  inv_involutive : Function.Involutive involution
  functionalEq : ∀ s, scatteringCoeff (involution s) = scatteringCoeff s

/-! ## Section 4: Functional equation and geostrophic balance -/

/-
The functional equation of the completed Riemann zeta function,
    expressed via the Kreĭn involution: ξ(1-s) = ξ(s).
-/
theorem krein_functionalEquation_completedZeta (s : ℂ) :
    completedRiemannZeta (kreinInvolution s) = completedRiemannZeta s := by
  rw [kreinInvolution_eq_functionalEqInvolution]
  exact functionalEquation_completedZeta s

/-- The canonical Geostrophic Fluid Dictionary instance for the
    completed Riemann zeta function. -/
noncomputable def zetaDictionary : GeostrophicFluidDictionary where
  involution := kreinInvolution
  scatteringCoeff := completedRiemannZeta
  inv_involutive := kreinInvolution_involutive
  functionalEq := krein_functionalEquation_completedZeta

/-- The commutator condition [J, Ω] = 0, abstractly. -/
def commutatorVanishes (d : GeostrophicFluidDictionary) : Prop :=
  ∀ s, d.scatteringCoeff (d.involution s) = d.scatteringCoeff s

/-- Geostrophic balance: [J, Ω] = 0 for the zeta dictionary.
    This is the spectral analogue of the Taylor–Proudman theorem. -/
theorem commutator_vanishes : commutatorVanishes zetaDictionary :=
  zetaDictionary.functionalEq

/-! ## Section 5: Zero Symmetry -/

/-
Zero symmetry: if ξ(s) = 0 then ξ(1-s) = 0.
-/
theorem zero_symmetry (s : ℂ) (h : completedRiemannZeta s = 0) :
    completedRiemannZeta (kreinInvolution s) = 0 := by
  rw [krein_functionalEquation_completedZeta, h]

/-
Abstract Taylor–Proudman spectral theorem: for any geostrophic
    dictionary, zeros of the scattering coefficient are J-symmetric.
-/
theorem taylorProudman_spectral (d : GeostrophicFluidDictionary) (s : ℂ)
    (h : d.scatteringCoeff s = 0) :
    d.scatteringCoeff (d.involution s) = 0 := by
  rw [d.functionalEq, h]

/-! ## Section 6: Geostrophic Pinning -/

/-
Geostrophic pinning: if J(s) = s then Re(s) = 1/2.
-/
theorem geostrophic_pinning (s : ℂ) (h : kreinInvolution s = s) :
    OnCriticalLine s := by
  unfold kreinInvolution at h
  unfold OnCriticalLine; norm_num [show s = 1 / 2 by linear_combination -h / 2]

/-
Off-critical zero has distinct partner.
-/
theorem off_critical_zero_has_partner (s : ℂ)
    (hz : completedRiemannZeta s = 0)
    (hoff : ¬ OnCriticalLine s) :
    kreinInvolution s ≠ s ∧ completedRiemannZeta (kreinInvolution s) = 0 := by
  refine ⟨fun h => hoff (geostrophic_pinning s h), zero_symmetry s hz⟩

/-! ## Section 7: Rigidity Theorem (conditional on RH) -/

/-
Under the Riemann Hypothesis, every non-trivial zero of the Riemann
    zeta function lies on the critical line.
-/
theorem RH_implies_all_zeros_on_axis
    (hRH : RiemannHypothesis)
    (s : ℂ) (hz : riemannZeta s = 0)
    (hnt : ¬∃ n : ℕ, s = -2 * (↑n + 1))
    (hne : s ≠ 1) :
    OnCriticalLine s := by
  have := @hRH s
  exact this hz hnt hne

/-
Under RH, the J-partner of any non-trivial zero is also on the
    critical line.
-/
theorem RH_partner_on_axis
    (hRH : RiemannHypothesis)
    (s : ℂ) (hz : riemannZeta s = 0)
    (hnt : ¬∃ n : ℕ, s = -2 * (↑n + 1))
    (hne : s ≠ 1) :
    OnCriticalLine (kreinInvolution s) := by
  apply kreinInvolution_preserves_criticalLine
  exact RH_implies_all_zeros_on_axis hRH s hz hnt hne

/-! ## Section 8: Odd Symmetry Master Link -/

/-
The Odd Symmetry Master Link: for any J-invariant differentiable
    nonvanishing function f, the derivative satisfies f'(J(s)) = -f'(s).
-/
theorem odd_symmetry_from_invariance
    (f : ℂ → ℂ) (f' : ℂ → ℂ)
    (hJ : ∀ s, f (kreinInvolution s) = f s)
    (hd : ∀ s, HasDerivAt f (f' s) s)
    (s : ℂ) :
    f' (kreinInvolution s) = - f' s := by
  have h_chain : deriv (fun s => f (1 - s)) s = deriv f (1 - s) * (-1) := by
    convert HasDerivAt.deriv (HasDerivAt.comp s (hd _) (hasDerivAt_id' s |> HasDerivAt.const_sub _)) using 1; ring!
    rw [hd _ |> HasDerivAt.deriv]
  simp_all +decide [kreinInvolution, hd _ |> HasDerivAt.deriv]

/-
Corollary: the logarithmic derivative (f'/f) is odd about the critical line.
-/
theorem logDeriv_odd_from_invariance
    (f : ℂ → ℂ) (f' : ℂ → ℂ)
    (hJ : ∀ s, f (kreinInvolution s) = f s)
    (hd : ∀ s, HasDerivAt f (f' s) s)
    (_hne : ∀ s, f s ≠ 0)
    (s : ℂ) :
    f' (kreinInvolution s) / f (kreinInvolution s) =
    - (f' s / f s) := by
  rw [← neg_div, ← odd_symmetry_from_invariance f f' hJ hd s]
  rw [hJ]

/-
The real part of the logarithmic derivative is antisymmetric
    about the critical line.
-/
theorem re_logDeriv_antisymmetric
    (f : ℂ → ℂ) (f' : ℂ → ℂ)
    (hJ : ∀ s, f (kreinInvolution s) = f s)
    (hd : ∀ s, HasDerivAt f (f' s) s)
    (hne : ∀ s, f s ≠ 0)
    (s : ℂ) :
    (f' (kreinInvolution s) / f (kreinInvolution s)).re =
    - (f' s / f s).re := by
  convert congr_arg Complex.re (logDeriv_odd_from_invariance f f' hJ hd hne s) using 1

/-- CJ-18: Geostrophic symmetry / odd symmetry from invariance. -/
def buchanan_geostrophic_symmetry := @odd_symmetry_from_invariance

/-- CJ-19: Taylor–Proudman spectral theorem. -/
def buchanan_taylor_proudman := @taylorProudman_spectral

end MNZI
