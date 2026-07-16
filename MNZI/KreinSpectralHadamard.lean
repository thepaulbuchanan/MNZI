/-
  MNZI/KreinSpectralHadamard.lean

  Kreĭn-Spectral Characterisation of Hadamard States
  for the Dirac Field on Globally Hyperbolic Spacetimes

  Paper C-2: Algebraic core formalisation.

  This file formalises the algebraic results of Paper C-2:
  - §3: J-spectral projections and eigenspace swap
  - §4: Two-point function algebraic properties
  - §5: J-positivity (main algebraic result)
  - Open Question 1: Strict positivity (resolved negatively)
  - §6: κ-obstruction
  - §7: Stationary/ultrastatic coincidence (eigenspace swap)
-/

import Mathlib
import MNZI.Core

noncomputable section

open InnerProductSpace ComplexInnerProductSpace
open scoped ComplexInnerProductSpace

namespace MNZI

/-! ## §3: J-Spectral Projections

We work in a complex Hilbert space H with a fundamental symmetry J
(self-adjoint involution: J² = id, J* = J). The J-spectral projections are
P₊ = ½(I + J) and P₋ = ½(I - J).
-/

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H]

/-- The positive J-spectral projection P₊ = ½(I + J). -/
def Pplus (J : H →L[ℂ] H) : H →L[ℂ] H :=
  (1/2 : ℂ) • (ContinuousLinearMap.id ℂ H + J)

/-- The negative J-spectral projection P₋ = ½(I - J). -/
def Pminus (J : H →L[ℂ] H) : H →L[ℂ] H :=
  (1/2 : ℂ) • (ContinuousLinearMap.id ℂ H - J)

/-
P₊ + P₋ = I (partition of identity).
-/
theorem Pplus_add_Pminus (J : H →L[ℂ] H) :
    Pplus J + Pminus J = ContinuousLinearMap.id ℂ H := by
      unfold Pplus Pminus; ext; simp +decide ; ring;
      module

/-
J acts as +id on the range of P₊: J(P₊ x) = P₊ x when J² = id.
-/
theorem J_on_Pplus (J : H →L[ℂ] H) (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (x : H) : J (Pplus J x) = Pplus J x := by
      simp +decide [ Pplus, hJ2 ];
      rw [ add_comm, ← ContinuousLinearMap.comp_apply, hJ2, ContinuousLinearMap.id_apply ]

/-
J acts as -id on the range of P₋: J(P₋ x) = -P₋ x when J² = id.
-/
theorem J_on_Pminus (J : H →L[ℂ] H) (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (x : H) : J (Pminus J x) = -(Pminus J x) := by
      unfold Pminus;
      simp_all +decide [ ContinuousLinearMap.ext_iff ];
      rw [ ← smul_neg, neg_sub ]

/-
P₊ is idempotent when J² = id.
-/
theorem Pplus_idempotent (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H) :
    (Pplus J).comp (Pplus J) = Pplus J := by
      unfold Pplus; simp +decide [ hJ2 ] ; ring;
      module

/-
P₋ is idempotent when J² = id.
-/
theorem Pminus_idempotent (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H) :
    (Pminus J).comp (Pminus J) = Pminus J := by
      convert Pplus_idempotent ( -J ) _ using 1;
      · unfold Pplus Pminus; simp +decide [ ContinuousLinearMap.smul_apply, ContinuousLinearMap.sub_apply, ContinuousLinearMap.add_apply ] ;
        module;
      · unfold Pplus Pminus; ext; simp +decide [ sub_eq_add_neg ] ;
      · simp_all +decide [ ContinuousLinearMap.ext_iff ]

/-
P₊ P₋ = 0 (orthogonality of projections).
-/
theorem Pplus_Pminus_zero (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H) :
    (Pplus J).comp (Pminus J) = 0 := by
      unfold Pplus Pminus; simp +decide [ hJ2 ] ; ring;
      abel1

/-
P₋ P₊ = 0 (orthogonality of projections, other direction).
-/
theorem Pminus_Pplus_zero (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H) :
    (Pminus J).comp (Pplus J) = 0 := by
      ext x;
      simp_all +decide [ Pplus, Pminus, ContinuousLinearMap.ext_iff ];
      module

/-! ### Anticommutation and Eigenspace Swap

When {J, D} = 0 (J and D anticommute), the key structural result is that
J maps eigenspaces of D to eigenspaces with negated eigenvalue.
-/

/-
If {J, D} = 0 and D v = μ v, then D(J v) = (-μ)(J v).
This is the eigenspace swap: J maps the μ-eigenspace to the (-μ)-eigenspace.
-/
theorem anticommutation_eigenspace_swap
    (J D : H →L[ℂ] H)
    (hanti : J.comp D + D.comp J = 0)
    (v : H) (μ : ℂ) (hev : D v = μ • v) :
    D (J v) = (-μ) • (J v) := by
      replace hanti := congr_arg ( fun f => f v ) hanti ; simp_all +decide [ add_eq_zero_iff_eq_neg ]

/-
D maps P₊-eigenvectors to P₋-eigenvectors when {J, D} = 0.
This is the corrected theorem: p_≥ ≠ P₊ in general.
-/
theorem eigenvalue_projection_swap
    (J D : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (hanti : J.comp D + D.comp J = 0)
    (v : H) (hv : Pplus J v = v) :
    Pminus J (D v) = D v := by
      simp_all +decide [ mul_add, add_mul, ContinuousLinearMap.ext_iff ];
      simp_all +decide [ Pplus, Pminus, add_eq_zero_iff_eq_neg ];
      convert congr_arg ( D ) hv using 1 ; norm_num [ ← two_smul, smul_smul ]

/-! ## §4: The J-Spectral Two-Point Function

The two-point function Λ⁺_J(ψ,φ) = ⟨J P₊ ψ, φ⟩ = [P₊ ψ, φ]_J.
When J² = id, J P₊ = P₊ (from J_on_Pplus), so Λ⁺_J(ψ,φ) = ⟨P₊ ψ, φ⟩.
-/

/-- The J-spectral two-point function Λ⁺(ψ,φ) = ⟨J P₊ ψ, φ⟩. -/
def LambdaPlus (J : H →L[ℂ] H) (ψ φ : H) : ℂ :=
  @inner ℂ H _ (J (Pplus J ψ)) φ

/-- The J-spectral two-point function Λ⁻(ψ,φ) = ⟨J P₋ ψ, φ⟩. -/
def LambdaMinus (J : H →L[ℂ] H) (ψ φ : H) : ℂ :=
  @inner ℂ H _ (J (Pminus J ψ)) φ

/-
Λ⁺(ψ,φ) = ⟨P₊ ψ, φ⟩ when J² = id (since J P₊ = P₊).
-/
theorem LambdaPlus_eq_inner_Pplus (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (ψ φ : H) :
    LambdaPlus J ψ φ = @inner ℂ H _ (Pplus J ψ) φ := by
      convert congr_arg ( fun x => ⟪x, φ⟫ ) ( J_on_Pplus J hJ2 ψ ) using 1

/-
Λ⁺ + Λ⁻ = [·,·]_J (partition of identity for two-point functions).
-/
theorem twopoint_partition (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (ψ φ : H) :
    LambdaPlus J ψ φ + LambdaMinus J ψ φ = @inner ℂ H _ (J ψ) φ := by
      unfold LambdaPlus LambdaMinus;
      rw [ ← inner_add_left, ← map_add ];
      rw [ ← ContinuousLinearMap.add_apply, Pplus_add_Pminus ];
      rfl

/-! ## §5: J-Positivity (Main Algebraic Result)

The main result: Λ⁺(ψ,ψ).re ≥ 0.
-/

/-
⟨P₊ ψ, ψ⟩ = ‖P₊ ψ‖² when J² = id and J* = J.
This uses the self-adjoint idempotent structure.
-/
theorem inner_Pplus_self_eq_norm_sq (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (hJsa : ContinuousLinearMap.adjoint J = J)
    (ψ : H) :
    @inner ℂ H _ (Pplus J ψ) ψ = (‖Pplus J ψ‖ : ℝ) ^ 2 := by
      have h_decomp : ⟪(Pplus J) ψ, ψ⟫ = ⟪(Pplus J) ψ, (Pplus J) ψ⟫ + ⟪(Pplus J) ψ, (Pminus J) ψ⟫ := by
        rw [ ← inner_add_right ] ; congr ; simp +decide [ Pplus, Pminus ] ; ring;
        module;
      have h_cross_term : ⟪(Pplus J) ψ, (Pminus J) ψ⟫ = 0 := by
        have h_self_adjoint : ⟪(Pplus J) ψ, (Pminus J) ψ⟫ = ⟪ψ, (Pplus J) ((Pminus J) ψ)⟫ := by
          have h_self_adjoint : ContinuousLinearMap.adjoint (Pplus J) = Pplus J := by
            unfold Pplus;
            ext; simp [hJsa];
            congr! 1;
            · simp +decide [ ContinuousLinearMap.adjoint ];
              erw [ LinearIsometryEquiv.symm_apply_apply ] ; norm_num;
              erw [ Complex.conj_ofReal ] ; norm_num;
            · convert congr_arg ( fun f => ( 2⁻¹ : ℂ ) • f ) ( congr_arg ( fun f => f ‹_› ) hJsa ) using 1;
              simp +decide [ ContinuousLinearMap.adjoint ];
              erw [ Complex.conj_ofReal ] ; norm_num;
          rw [ ← ContinuousLinearMap.adjoint_inner_right, h_self_adjoint ];
        rw [ h_self_adjoint, show ( Pplus J ) ( ( Pminus J ) ψ ) = 0 from by simpa using congr_arg ( fun f => f ψ ) ( Pplus_Pminus_zero J hJ2 ) ] ; simp +decide;
      simp_all +decide [ inner_self_eq_norm_sq_to_K ]

/-
**Main algebraic result**: Λ⁺(ψ,ψ).re ≥ 0 (J-positivity).
This is Theorem 5.1(i) of Paper C-2.
-/
theorem j_positivity (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (hJsa : ContinuousLinearMap.adjoint J = J)
    (ψ : H) :
    0 ≤ (LambdaPlus J ψ ψ).re := by
      rw [ LambdaPlus_eq_inner_Pplus, inner_Pplus_self_eq_norm_sq ];
      · norm_cast ; positivity;
      · exact hJ2;
      · exact hJsa;
      · exact hJ2

/-
Λ⁺(ψ,ψ) = 0 ↔ P₊ ψ = 0.
-/
theorem LambdaPlus_self_eq_zero_iff (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (hJsa : ContinuousLinearMap.adjoint J = J)
    (ψ : H) :
    LambdaPlus J ψ ψ = 0 ↔ Pplus J ψ = 0 := by
      have h1 : LambdaPlus J ψ ψ = 0 ↔ inner ℂ (Pplus J ψ) ψ = 0 := by
        rw [ LambdaPlus_eq_inner_Pplus ];
        exact hJ2;
      rw [ h1, inner_Pplus_self_eq_norm_sq J hJ2 hJsa ];
      norm_num +zetaDelta at *

/-! ## Open Question 1: Strict Positivity (Resolved Negatively)

Strict positivity fails: there exist nonzero ψ with Λ⁺(ψ,ψ) = 0.
The correct characterisation is: Λ⁺(ψ,ψ).re > 0 ↔ P₊ ψ ≠ 0.
-/

/-
Strict positivity fails in the abstract setting:
any ψ ∈ range(P₋) with ψ ≠ 0 gives Λ⁺(ψ,ψ) = 0.
-/
theorem strict_positivity_fails_abstract
    (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (hJsa : ContinuousLinearMap.adjoint J = J)
    (ψ : H) (hψ : ψ ≠ 0) (hψminus : Pminus J ψ = ψ) :
    LambdaPlus J ψ ψ = 0 := by
      rw [ LambdaPlus_eq_inner_Pplus ];
      · have hPplus_zero : Pplus J ψ = ψ - Pminus J ψ := by
          simp +decide [ Pplus, Pminus ];
          module;
        aesop;
      · exact hJ2

/-
Complete characterisation: Λ⁺(ψ,ψ).re > 0 ↔ P₊ ψ ≠ 0.
-/
theorem strict_positivity_iff (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (hJsa : ContinuousLinearMap.adjoint J = J)
    (ψ : H) :
    0 < (LambdaPlus J ψ ψ).re ↔ Pplus J ψ ≠ 0 := by
      rw [ LambdaPlus_eq_inner_Pplus, inner_Pplus_self_eq_norm_sq ];
      · norm_cast ; aesop;
      · exact hJ2;
      · exact hJsa;
      · exact hJ2

/-! ## §6: The κ-Obstruction

The J-index κ measures the difference between the dimensions of the
J-positive and J-negative parts of the kernel. κ = 0 is necessary
for the J-spectral two-point function to be well-defined.
-/

/-- The κ-index: difference of dimensions of J-positive and J-negative
parts of a finite-dimensional kernel. -/
def kappa (dimPlus dimMinus : ℤ) : ℤ := dimPlus - dimMinus

/-- κ = 0 ↔ balanced dimensions. -/
theorem kappa_zero_iff (dp dm : ℤ) :
    kappa dp dm = 0 ↔ dp = dm := by
  simp [kappa, sub_eq_zero]

/-- In the Eisenstein model with equal dimensions, κ = 0. -/
theorem eisenstein_kappa_zero (d : ℤ) :
    kappa d d = 0 := by
  simp [kappa]

/-- κ is additive under direct sums. -/
theorem kappa_additive (dp1 dm1 dp2 dm2 : ℤ) :
    kappa (dp1 + dp2) (dm1 + dm2) = kappa dp1 dm1 + kappa dp2 dm2 := by
  simp [kappa]; ring

/-! ## §7: Stationary Coincidence

The anticommutation eigenspace swap (already proved above) is the
key algebraic result used in the stationary coincidence argument.
The theorem `anticommutation_eigenspace_swap` establishes that
{J, D} = 0 ∧ D v = μ v → D(J v) = (-μ)(J v).
-/

/-! ## Additional structural results -/

/-
P₊ ψ + P₋ ψ = ψ for any ψ (pointwise version of partition).
-/
theorem Pplus_add_Pminus_apply (J : H →L[ℂ] H) (ψ : H) :
    Pplus J ψ + Pminus J ψ = ψ := by
      convert congr_arg ( fun x => x ψ ) ( Pplus_add_Pminus J ) using 1

/-
P₊(P₊ ψ) = P₊ ψ (pointwise idempotency).
-/
theorem Pplus_idempotent_apply (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H) (ψ : H) :
    Pplus J (Pplus J ψ) = Pplus J ψ := by
      apply congr_arg (fun f => f ψ) (Pplus_idempotent J hJ2)

/-
P₋(P₋ ψ) = P₋ ψ (pointwise idempotency).
-/
omit [CompleteSpace H] in
theorem Pminus_idempotent_apply (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H) (ψ : H) :
    Pminus J (Pminus J ψ) = Pminus J ψ := by
      unfold Pminus; ring;
      simp_all +decide [ ContinuousLinearMap.ext_iff ];
      module

/-
P₊(P₋ ψ) = 0 (pointwise orthogonality).
-/
theorem Pplus_Pminus_zero_apply (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H) (ψ : H) :
    Pplus J (Pminus J ψ) = 0 := by
      grind +suggestions

/-
Λ⁺(ψ,φ) - Λ⁻(ψ,φ) = ⟨ψ, φ⟩ (commutator function).
This is Proposition 4.1(ii): since P₊ - P₋ = J, we get
J(P₊ - P₋) = J² = I, so the difference recovers the standard inner product.
-/
theorem twopoint_commutator (J : H →L[ℂ] H)
    (hJ2 : J.comp J = ContinuousLinearMap.id ℂ H)
    (ψ φ : H) :
    LambdaPlus J ψ φ - LambdaMinus J ψ φ = @inner ℂ H _ ψ φ := by
      have hJ_diff : J (Pplus J ψ) - J (Pminus J ψ) = ψ := by
        grind +suggestions;
      convert congr_arg ( fun x => inner ℂ x φ ) hJ_diff using 1;
      unfold LambdaPlus LambdaMinus; simp +decide [ inner_sub_left ] ;

end MNZI
