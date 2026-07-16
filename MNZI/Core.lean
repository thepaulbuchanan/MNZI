/-
  MNZI/Core.lean

  The canonical shared definitions for the MNZI programme.
  All definitions and theorems here are the single authoritative
  versions; individual paper files import this file and remove
  their local copies.

  Phase 2 (Definition Unification) — Session 29, 31 May 2026.

  Contents:
    §1  The Coil Invariant (π²/12)                   [Priority 1]
    §2  The Golden Ratio                              [Priority 2]
    §3  Gap Constants and Exponents                   [Priority 4]
    §4  The Criticality Sum                           [Priority 5]
    §5  The Critical Line Predicate                   [lower priority]
    §6  The Functional Equation Involution            [lower priority]
    §7  FundamentalSymmetry (hierarchy)               [Priority 3 — skeleton]
    §8  Placeholder for CJ Alias Declarations         [Phase 4, Session 30]

  Usage: at the top of each affected paper file, add:
    import MNZI.Core
  and remove the corresponding local definition.

  Naming convention: all names match the canonical names confirmed
  in the Session 28 journal (Part 6). The buchanan_* alias
  declarations will be added in Phase 4 (Session 30).
-/

import Mathlib
import MNZI.PurityReductionConfinement

namespace MNZI

open Real Complex

noncomputable section

/-! ## §1: The Coil Invariant (π²/12)

The Buchanan Coil Invariant: π²/12 = ζ(2)/2.
Canonical name and definition from Paper S (ZeroDensityConjectureCurvature).

Appears under various names across 10 files:
  coilInvariant     (S)
  coilEnergy        (T, Q)
  springConstantValue (M)
  coil_invariant_basel / coil_exceeds_landauer (N, R, V, D-1, L, U — inline)
-/

/-- The Buchanan Coil Invariant: π²/12 = ζ(2)/2.
    This constant appears throughout the MNZI programme as the natural
    spectral energy scale, equalling the Basel sum ∑ 1/n² divided by 2. -/
noncomputable def coilInvariant : ℝ := π ^ 2 / 12

/-- The coil invariant is positive. -/
theorem coilInvariant_pos : 0 < coilInvariant := by
  unfold coilInvariant; positivity

/-- The coil invariant equals ζ(2)/2 = (π²/6)/2. -/
theorem coilInvariant_eq_zeta2_div2 : coilInvariant = (π ^ 2 / 6) / 2 := by
  simp only [coilInvariant]; ring

/-- The coil invariant exceeds the Landauer erasure bound ln 2.
    Numerically: π²/12 ≈ 0.8225 > 0.6931 ≈ ln 2.
    Proof uses Real.quadratic_le_exp_of_nonneg (Paper U's approach). -/
theorem coilInvariant_exceeds_landauer : Real.log 2 < coilInvariant := by
  unfold coilInvariant
  have hexp34 : (2 : ℝ) < Real.exp (3 / 4) := by
    calc (2 : ℝ) < 1 + 3 / 4 + (3 / 4) ^ 2 / 2 := by norm_num
      _ ≤ Real.exp (3 / 4) :=
          Real.quadratic_le_exp_of_nonneg (by norm_num : (0 : ℝ) ≤ 3 / 4)
  have hlog : Real.log 2 < 3 / 4 := by
    rw [show (3 : ℝ) / 4 = Real.log (Real.exp (3 / 4)) from
        (Real.log_exp (3 / 4)).symm]
    exact Real.log_lt_log (by positivity) hexp34
  have hpi : (3 : ℝ) < π := pi_gt_three
  nlinarith [sq_nonneg π]

/-- The coil invariant is at least 3/4. -/
theorem coilInvariant_gt_three_quarters : (3 : ℝ) / 4 < coilInvariant := by
  unfold coilInvariant
  have hpi : (3 : ℝ) < π := pi_gt_three
  nlinarith [sq_nonneg π]


/-! ## §2: The Golden Ratio

Canonical form: `Real.goldenRatio` (Mathlib).
For φ⁻¹, use `Real.goldenRatio⁻¹` or the alias below.

Files using local φ definitions to be migrated:
  A   (goldenRatioInv)
  A-1 (phiInv)
  B   (local φ)
  B-1 (local φ)
  B-2 (local φ)
  H   (phi_inv — already uses Real.goldenRatio for φ itself)

Files already using canonical form: H, J, K, T.
-/

/-- The golden ratio inverse φ⁻¹ = (√5 - 1)/2.
    Alias for Real.goldenRatio⁻¹ provided for convenience. -/
noncomputable def goldenRatioInv : ℝ := Real.goldenRatio⁻¹

/-- φ⁻¹ is positive. -/
theorem goldenRatioInv_pos : 0 < goldenRatioInv :=
  inv_pos.mpr Real.goldenRatio_pos

/-
φ⁻¹ ∈ (0, 1).
-/
theorem goldenRatioInv_lt_one : goldenRatioInv < 1 := by
  exact inv_lt_one_of_one_lt₀ <| by ring_nf; nlinarith [ Real.sqrt_nonneg 5, Real.sq_sqrt ( show 0 ≤ 5 by norm_num ) ] ;

/-- φ² = φ + 1 (Mathlib's goldenRatio_sq). -/
theorem goldenRatio_sq : Real.goldenRatio ^ 2 = Real.goldenRatio + 1 :=
  Real.goldenRatio_sq

/-
φ⁻¹ = (√5 - 1)/2.
-/
theorem goldenRatioInv_explicit :
    goldenRatioInv = (Real.sqrt 5 - 1) / 2 := by
      rw [ show goldenRatioInv = ( ( 1 + Real.sqrt 5 ) / 2 ) ⁻¹ from rfl, inv_eq_of_mul_eq_one_right ] ; ring ; norm_num;

/-
φ⁻¹² + φ⁻¹ - 1 = 0, the minimal polynomial of φ⁻¹.
-/
theorem goldenRatioInv_sq_add : goldenRatioInv ^ 2 + goldenRatioInv - 1 = 0 := by
  unfold goldenRatioInv;
  grind +qlia

/-! ## §3: Gap Constants and Exponents

Canonical functional forms from Paper S (ZeroDensityConjectureCurvature).
Constant forms (for L, N, U) derived from these.

Files to update: L (EnergyDeviationGap), N (ZetaVorticityWeb), U (MagneticInterpretationGapStrip).
Paper S already uses these names.
-/

/-- The gap width as a function of the density exponent A:
    gapWidthFn(A) = (A - 2) / (2A) = 1/2 - 1/A.
    This is the functional form from Paper S.
    The name gapWidthFn distinguishes it from the constant gapWidth below. -/
noncomputable def gapWidthFn (A : ℝ) : ℝ := (A - 2) / (2 * A)

/-- The gap boundary as a function of A: 1 - 1/A. -/
noncomputable def gapBoundaryFn (A : ℝ) : ℝ := 1 - 1 / A

/-- gapWidthFn(A) = 1/2 - 1/A when A ≠ 0. -/
theorem gapWidthFn_eq_half_minus_inv (A : ℝ) (hA : A ≠ 0) :
    gapWidthFn A = 1 / 2 - 1 / A := by
  unfold gapWidthFn; field_simp

/-- gapBoundaryFn(A) = 1/2 + gapWidthFn(A) when A ≠ 0. -/
theorem gapBoundaryFn_eq_half_plus_gapWidthFn (A : ℝ) (hA : A ≠ 0) :
    gapBoundaryFn A = 1 / 2 + gapWidthFn A := by
  unfold gapBoundaryFn gapWidthFn; field_simp; ring

/-- The Heath-Brown density exponent: A = 12/5. -/
def heathBrownExponent : ℝ := 12 / 5

/-- The Guth-Maynard density exponent: A = 30/13. -/
def guthMaynardExponent : ℝ := 30 / 13

/-- The Zero Density Conjecture exponent: A = 2. -/
def zdcExponent : ℝ := 2

/-- The Guth-Maynard gap boundary constant: σ_max = 17/30. -/
def gapHi : ℝ := 17 / 30

/-- The Guth-Maynard gap width constant: 1/15. -/
def gapWidth : ℝ := 1 / 15

/-- gapHi = gapBoundaryFn(guthMaynardExponent). -/
theorem gapHi_eq : gapHi = gapBoundaryFn guthMaynardExponent := by
  unfold gapHi gapBoundaryFn guthMaynardExponent; norm_num

/-- gapWidth = gapWidthFn(guthMaynardExponent). -/
theorem gapWidth_eq : gapWidth = gapWidthFn guthMaynardExponent := by
  unfold gapWidth gapWidthFn guthMaynardExponent; norm_num

/-- gapHi > 1/2. -/
theorem gapHi_gt_half : (1 : ℝ) / 2 < gapHi := by
  unfold gapHi; norm_num

/-- gapHi < 1. -/
theorem gapHi_lt_one : gapHi < 1 := by
  unfold gapHi; norm_num

/-- gapWidth > 0. -/
theorem gapWidth_pos : (0 : ℝ) < gapWidth := by
  unfold gapWidth; norm_num

/-- Gap width at Heath-Brown exponent = 1/12. -/
theorem gapWidthFn_heathBrown : gapWidthFn heathBrownExponent = 1 / 12 := by
  unfold gapWidthFn heathBrownExponent; norm_num

/-- Gap width at Guth-Maynard exponent = 1/15. -/
theorem gapWidthFn_guthMaynard : gapWidthFn guthMaynardExponent = 1 / 15 := by
  unfold gapWidthFn guthMaynardExponent; norm_num

/-- Gap boundary at Guth-Maynard exponent = 17/30. -/
theorem gapBoundaryFn_guthMaynard : gapBoundaryFn guthMaynardExponent = 17 / 30 := by
  unfold gapBoundaryFn guthMaynardExponent; norm_num


/-! ## §4: The Criticality Sum

`criticalitySum x n = ∑_{k=0}^{n-1} x^(k+1)` — canonical name from Paper T.
Identical to `recipSum r n` in Paper P (AdelicModeConvergence).
Paper P should be updated to use this name (or alias recipSum := criticalitySum).
-/

/-- The criticality sum: ∑_{k=0}^{n-1} x^(k+1).
    This is the n-bonacci criticality condition sum (Paper T)
    and the adelic partial sum (Paper P, as `recipSum`). -/
noncomputable def criticalitySum (x : ℝ) (n : ℕ) : ℝ :=
  ∑ k ∈ Finset.range n, x ^ (k + 1)

/-
Closed form: criticalitySum x n = x(1 - xⁿ)/(1 - x) for x ≠ 1.
-/
theorem criticalitySum_formula {x : ℝ} (hx : x ≠ 1) (n : ℕ) :
    criticalitySum x n = x * (1 - x ^ n) / (1 - x) := by
      rw [ eq_div_iff ( sub_ne_zero_of_ne hx.symm ) ] ; induction' n with n ih <;> simp_all +decide [ pow_succ', Finset.sum_range_succ ] ; ring;
      · exact Or.inl rfl;
      · unfold criticalitySum at *; norm_num [ Finset.sum_range_succ ] at *; linear_combination ih;

/-
criticalitySum(1/2, n) = 1 - (1/2)^n.
-/
theorem criticalitySum_half (n : ℕ) :
    criticalitySum (1 / 2) n = 1 - (1 / 2) ^ n := by
      convert criticalitySum_formula _ _ using 1 <;> norm_num

/-
criticalitySum(1/2, n) < 1 for all n ≥ 1.
-/
theorem criticalitySum_half_lt_one (n : ℕ) :
    criticalitySum (1 / 2) n < 1 := by
      rw [ criticalitySum_half ] ; exact sub_lt_self _ ( by positivity )

/-- criticalitySum is continuous in x for fixed n. -/
theorem criticalitySum_continuous (n : ℕ) :
    Continuous (fun x : ℝ => criticalitySum x n) := by
  unfold criticalitySum
  apply continuous_finset_sum
  intro k _
  exact continuous_pow (k + 1)

/-- The adelic reciprocal sum alias: recipSum = criticalitySum.
    Paper P uses the name `recipSum`; this alias maintains compatibility. -/
noncomputable def recipSum (r : ℝ) (n : ℕ) : ℝ := criticalitySum r n

/-- recipSum(1/2, n) → 1 as n → ∞. -/
theorem recipSum_half_tendsto :
    Filter.Tendsto (fun n => recipSum (1 / 2) n) Filter.atTop (nhds 1) := by
  unfold recipSum
  have : Filter.Tendsto (fun n => criticalitySum (1 / 2) n) Filter.atTop (nhds 1) := by
    have : Filter.Tendsto (fun n => (1 : ℝ) - (1 / 2) ^ n) Filter.atTop (nhds 1) := by
      have h : Filter.Tendsto (fun n : ℕ => (1 / 2 : ℝ) ^ n) Filter.atTop (nhds 0) :=
        tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
      simpa using h.const_sub 1
    simp_rw [criticalitySum_half] at *
    exact this
  exact this


/-! ## §5: The Critical Line Predicate

OnCriticalLine s ↔ s.re = 1/2.
Defined in Paper O (GeostrophicRigidity); also implicit in F, I, V.
-/

/-- A complex number is on the critical line if its real part is 1/2. -/
def OnCriticalLine (s : ℂ) : Prop := s.re = 1 / 2

/-- 1/2 + it is on the critical line for all t. -/
theorem OnCriticalLine_half_add_I_mul (t : ℝ) :
    OnCriticalLine ((1 / 2 : ℝ) + ↑t * Complex.I) := by
  unfold OnCriticalLine
  simp [Complex.add_re, Complex.mul_re, Complex.I_re, Complex.I_im]

/-- If s is on the critical line then so is 1 - s. -/
theorem OnCriticalLine_one_sub {s : ℂ} (h : OnCriticalLine s) :
    OnCriticalLine (1 - s) := by
  unfold OnCriticalLine at *
  simp [Complex.sub_re]
  linarith


/-! ## §6: The Functional Equation Involution

Two distinct uses of "kreinInvolution" in the programme:
  (a) s ↦ 1-s  (Papers O, V — the functional equation map on ℂ)
  (b) Abstract involutory operator J (Papers C, D, D-1 — the Kreĭn symmetry)

Resolution per journal Part 6:
  `functionalEqInvolution` for s ↦ 1-s (this section)
  `kreinInvolution` reserved for operator J (§7 below)
  `involutionS` for s ↦ 1-conj(s) (Paper M) — already distinct

Papers O and V currently define `kreinInvolution` as s ↦ 1-s.
After Phase 2 migration, they will use `functionalEqInvolution`.
-/

/-- The functional equation involution on ℂ: s ↦ 1 - s. -/
def functionalEqInvolution (s : ℂ) : ℂ := 1 - s

/-- functionalEqInvolution is an involution. -/
theorem functionalEqInvolution_involutive :
    Function.Involutive functionalEqInvolution :=
  fun s => by unfold functionalEqInvolution; ring

/-- functionalEqInvolution is bijective. -/
theorem functionalEqInvolution_bijective :
    Function.Bijective functionalEqInvolution :=
  functionalEqInvolution_involutive.bijective

/-- The completed Riemann zeta function satisfies the functional equation. -/
theorem functionalEquation_completedZeta (s : ℂ) :
    completedRiemannZeta (functionalEqInvolution s) = completedRiemannZeta s := by
  unfold functionalEqInvolution
  exact completedRiemannZeta_one_sub s

/- The original statement "fixed points ↔ OnCriticalLine" is false:
   1 - s = s forces s = 1/2 (a single point), but OnCriticalLine
   only requires s.re = 1/2 (the entire critical line).
   Corrected: functionalEqInvolution preserves the critical line.
   DISPROVED: counterexample s = 1/2 + i/2 is OnCriticalLine but not a fixed point.
   -- theorem functionalEqInvolution_fixed_iff (s : ℂ) :
   --     functionalEqInvolution s = s ↔ OnCriticalLine s := ...
-/

/-- functionalEqInvolution maps the critical line to itself. -/
theorem functionalEqInvolution_preserves_criticalLine {s : ℂ}
    (h : OnCriticalLine s) : OnCriticalLine (functionalEqInvolution s) := by
  exact OnCriticalLine_one_sub h

/-
The unique fixed point of functionalEqInvolution is s = 1/2.
-/
theorem functionalEqInvolution_fixed_iff (s : ℂ) :
    functionalEqInvolution s = s ↔ s = 1 / 2 := by
      grind +locals

/-! ## §7: FundamentalSymmetry Hierarchy (Skeleton)

Six variants of FundamentalSymmetry found across the programme:
  C   ContinuousLinearMap — J, sq_eq_id, adj
  C-1 Abstract ring element — J, J_sq
  C-3 ContinuousLinearMap — J, sq_eq_id, adj_eq
  D   LinearMap (module)  — J, J_sq
  D-1 AddMonoidHom        — J, sq_eq_id
  D-2 Concrete matrix     — τz

Strategy: hierarchy with three levels.
NOTE: Full unification of FundamentalSymmetry is the most complex
Phase 2 target and is deferred if Session 29 time runs short
(per Part 14 of the journal). The type class hierarchy below is
the scaffold; individual files will be migrated to use it in
subsequent sessions.

For now, each paper's local FundamentalSymmetry definition remains.
The scaffold here records the intended structure.
-/

-- Level 0: Abstract ring version (most general, from C-1 style)
/-- Level 0: A fundamental symmetry element in a ring.
    J satisfies J² = 1. -/
structure FundSymm0 (R : Type*) [Ring R] where
  J : R
  J_sq : J * J = 1

/-- J is self-inverse at Level 0. -/
theorem FundSymm0.J_inv {R : Type*} [Ring R] (fs : FundSymm0 R) :
    fs.J * fs.J = 1 := fs.J_sq

-- Level 1: Module version (D style) — adds linearity
/-- Level 1: A fundamental symmetry as a linear map on a module.
    J : M →ₗ[R] M satisfies J ∘ J = id. -/
structure FundSymm1 (R : Type*) (M : Type*) [CommRing R]
    [AddCommGroup M] [Module R M] where
  J : M →ₗ[R] M
  J_sq : J.comp J = LinearMap.id

/-- J is injective at Level 1. -/
theorem FundSymm1.J_injective {R M : Type*} [CommRing R]
    [AddCommGroup M] [Module R M] (fs : FundSymm1 R M) :
    Function.Injective fs.J := by
  intro x y hxy
  have hx := congr_fun (congr_arg DFunLike.coe fs.J_sq) x
  have hy := congr_fun (congr_arg DFunLike.coe fs.J_sq) y
  simp [LinearMap.comp_apply] at hx hy
  rw [← hx, ← hy, hxy]

-- Level 2: Hilbert space version (C-3 style) — adds adjoint
-- (requires NormedAddCommGroup, InnerProductSpace, CompleteSpace)
-- Defined in individual papers; migration to Core in a later session.


/-! ## §8: CJ Alias Declarations (Phase 5 — activated)

All 27 buchanan_* alias declarations are now live Lean definitions.
Each alias is placed in the paper file containing its target theorem
(to avoid circular imports and cross-file name conflicts).

Summary of alias locations:

| CJ# | Alias name                          | Target                              | File                                      |
|------|---------------------------------------|---------------------------------------|-------------------------------------------|
| 01   | buchanan_atas_identity              | algebraic_identity                  | FiniteOrderCorrections.lean               |
| 02   | buchanan_gue_mode                   | gue_mode_is_golden_ratio_inv        | GoldenRatioGUEMode.lean                   |
| 03   | buchanan_golden_straddle            | incommensurability                  | PythagoreanGoldenDuality.lean             |
| 04   | buchanan_aps_formula                | krein_aps_master                    | KreinAPSFormula.lean                      |
| 05   | buchanan_seeley_anomaly             | vanishing_four_way                  | KreinIndexSkeleton.lean                   |
| 06   | buchanan_connes_unification         | full_equivalence_chain              | ConnesKreinUnification.lean               |
| 07   | buchanan_krein_string               | ternary_equivalence                 | KreinFellerString.lean                    |
| 08   | buchanan_topological_protection     | link_h                              | CyclicCocycleAnnihilation.lean            |
| 09   | buchanan_golden_bridge              | masterLink                          | ArithmeticDysonGas.lean                   |
| 10   | buchanan_odd_symmetry               | re_logDeriv_antisymmetric           | RiemannXiOddSymmetry.lean                 |
| 11   | buchanan_equivalence_chain          | ElevenLinkChain (type alias)        | RiemannXiOddSymmetry.lean                 |
| 12   | buchanan_goldbach_wastlund          | golden_identity_2                   | PrimeShadeDuality.lean                    |
| 13   | buchanan_pythagorean_density        | master_claim                        | MusicalArchitecturePrimes.lean            |
| 14   | buchanan_energy_reinforcement       | no_zeros_in_gap…                    | EnergyDeviationGap.lean                   |
| 15   | buchanan_barycentre                 | barycentre_re_half                  | EquatorialFixedPointGeometry.lean         |
| 16   | buchanan_equatorial_fixed_point     | involutionS_fixed_iff               | EquatorialFixedPointGeometry.lean         |
| 17   | buchanan_ghb_condition              | fold_anchor_unique                  | ZetaVorticityWeb.lean                     |
| 18   | buchanan_geostrophic_symmetry       | odd_symmetry_from_invariance        | GeostrophicRigidity.lean                  |
| 19   | buchanan_taylor_proudman            | taylorProudman_spectral             | GeostrophicRigidity.lean                  |
| 20   | buchanan_adelic_convergence         | recipSum_half_tendsto               | Core.lean (this file, below)              |
| 21   | buchanan_filatov_fold               | (self-contained)                    | TwoQubitEntanglement.lean                 |
| 22   | buchanan_zdc_gauss_bonnet           | paper_S_summary                     | ZeroDensityConjectureCurvature.lean       |
| 23   | buchanan_reflective_shade           | reflective_shade_duality            | GeometricRealizationHB.lean               |
| 24   | buchanan_lorentzian_phase_involution| scattering_phase_involution         | NaturalKreinLorentzianDirac.lean          |
| 25   | buchanan_j_triple_unification       | axioms_consistent                   | KreinSpectralTriples.lean                 |
| 26   | buchanan_kitaev_bbc                 | kitaev_bbc                          | KreinKitaevJAPSFormula_revised.lean       |
| 27   | buchanan_derivative_tower           | phi_prime_unit_modulus               | DerivativeTowerRiemannXiFunction_revised  |
| 28   | buchanan_confinement_reduction      | confinement_exact                   | PurityReductionConfinement.lean           |
| 29   | buchanan_golden_straddle_certified  | straddle_two_thirds                 | GoldenPrime.lean                          |

All 29 aliases verified sorry-free.
Note: CJ-29 additionally uses Lean.ofReduceBool and
Lean.trustCompiler (via native_decide). All other aliases
use propext, Classical.choice, Quot.sound only.

Note on name conflicts: cross-file name conflicts in the MNZI namespace
(~60 duplicate names across the 31 files) prevent a single-file import.
Aliases are therefore co-located with their targets. The conflicts are
benign — each file builds independently — but a future unification pass
(Phase 6) should deduplicate shared helper lemmas into Core.lean.
-/


/-- CJ-20: Adelic convergence. -/
theorem buchanan_adelic_convergence :
    Filter.Tendsto (fun n => recipSum (1 / 2) n) Filter.atTop (nhds 1) :=
  recipSum_half_tendsto

/-- CJ-28: Confinement Reduction. The algebraic characterisation of exact
    confinement: φ^(k) ∈ {+i, −1} at a Riemann zero is equivalent to
    z/(−conj z) ∈ {I, −1}, which holds iff z.re + z.im = 0 or z.im = 0.
    Source: PurityReductionConfinement.lean (Paper R-2). -/
theorem buchanan_confinement_reduction (z : ℂ) (hz : z ≠ 0) :
    z / (-(starRingEnd ℂ z)) = Complex.I ∨ z / (-(starRingEnd ℂ z)) = -1 ↔
    z.re + z.im = 0 ∨ z.im = 0 :=
  confinement_exact z hz

end  -- noncomputable section

end MNZI