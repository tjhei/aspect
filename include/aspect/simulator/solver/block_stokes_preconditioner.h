/*
  Copyright (C) 2011 - 2025 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#ifndef _aspect_block_stokes_preconditioner_h
#define _aspect_block_stokes_preconditioner_h

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>


#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>

namespace aspect
{

  namespace internal
  {
    /**
      * This class is used in the implementation of the right preconditioner
      * as an approximation for the inverse of the velocity (A) block.
      * This operator can either just apply the preconditioner (AMG)
      * or perform an inner CG / BiCGStab solve with the same preconditioner.
      */
    template <class PreconditionerA, class VectorType, class ABlockType>
    class InverseVelocityBlock
    {
      public:
        /**
         * Constructor.
         * @param matrix The matrix that contains A (from the system matrix)
         * @param preconditioner The preconditioner to be used
         * @param do_solve_A A flag indicating whether we should actually solve with
         *     the matrix $A$, or only apply one preconditioner step with it.
         * @param A_block_is_symmetric A flag indicating whether the matrix $A$ is symmetric.
         * @param solver_tolerance The tolerance for the CG solver which computes
         *     the inverse of the A block.
         */
        InverseVelocityBlock(const ABlockType &matrix,
                             const PreconditionerA &preconditioner,
                             const bool do_solve_A,
                             const bool A_block_is_symmetric,
                             const double solver_tolerance);

        void vmult(VectorType &dst,
                   const VectorType &src) const;

        unsigned int n_iterations() const;

      private:
        mutable unsigned int n_iterations_;
        const ABlockType &matrix;
        const PreconditionerA &preconditioner;
        const bool do_solve_A;
        const bool A_block_is_symmetric;
        const double solver_tolerance;
    };



    template <class PreconditionerA,class VectorType, class ABlockType>
    InverseVelocityBlock<PreconditionerA,VectorType,ABlockType>::InverseVelocityBlock(
      const ABlockType &matrix,
      const PreconditionerA &preconditioner,
      const bool do_solve_A,
      const bool A_block_is_symmetric,
      const double solver_tolerance)
      : n_iterations_ (0),
        matrix (matrix),
        preconditioner (preconditioner),
        do_solve_A (do_solve_A),
        A_block_is_symmetric (A_block_is_symmetric),
        solver_tolerance (solver_tolerance)
    {}



    /**
    * Implements the vmult for InverseVelocityBlock. This applies the action of A^{-1} by either
    * performing a solve with A or using a preconditioner sweep.
    */
    template <class PreconditionerA, class VectorType, class ABlockType>
    void InverseVelocityBlock<PreconditionerA,VectorType,ABlockType>::vmult(VectorType &dst,
                                                                            const VectorType &src) const
    {

      // Either solve with the top left block
      // or just apply one preconditioner sweep (for the first few
      // iterations of our two-stage outer GMRES iteration)
      if (do_solve_A == true)
        {
          SolverControl solver_control(10000, src.l2_norm() * solver_tolerance);
          PrimitiveVectorMemory<VectorType> mem;

          try
            {
              dst = 0.0;

              if (A_block_is_symmetric)
                {
                  SolverCG<VectorType> solver(solver_control, mem);
                  solver.solve(matrix, dst, src, preconditioner);
                }
              else
                {
                  // Use BiCGStab for non-symmetric matrices.
                  // BiCGStab can also solve indefinite systems if necessary.
                  // Do not compute the exact residual, as this
                  // is more expensive, and we only need an approximate solution.
                  SolverBicgstab<VectorType>
                  solver(solver_control,
                         mem,
                         typename SolverBicgstab<VectorType>::AdditionalData(/*exact_residual=*/ false));
                  solver.solve(matrix, dst, src, preconditioner);
                }
              n_iterations_ += solver_control.last_step();
            }
          catch (const std::exception &exc)
            {
              // if the solver fails, report the error from processor 0 with some additional
              // information about its location, and throw a quiet exception on all other
              // processors
              Utilities::throw_linear_solver_failure_exception("iterative (top left) solver",
                                                               "BlockSchurPreconditioner::vmult",
                                                               std::vector<SolverControl> {solver_control},
                                                               exc,
                                                               src.get_mpi_communicator());
            }
        }
      else
        {
          preconditioner.vmult (dst, src);
          n_iterations_ += 1;
        }
    }



    template <class PreconditionerA, class VectorType, class ABlockType>
    unsigned int InverseVelocityBlock<PreconditionerA, VectorType, ABlockType>::n_iterations() const
    {
      return n_iterations_;
    }
    /**
    * Base class for Schur Complement operators.
    */
    template<class VectorType>
    class SchurComplementOperator
    {
      public:
        virtual ~SchurComplementOperator() = default;

        virtual void vmult(VectorType &dst,
                           const VectorType &src) const=0;
        virtual unsigned int n_iterations() const=0;

    };



    template <class AInvOperator, class BTOperator, class VectorType, class PressureVectorType>
    class BlockSchurPreconditioner : public
#if DEAL_II_VERSION_GTE(9,7,0)
      EnableObserverPointer
#else
      Subscriptor
#endif
    {
      public:
        BlockSchurPreconditioner(
          const AInvOperator                                  &A_inverse_operator,
          const SchurComplementOperator<PressureVectorType>   &S_inverse_operator,
          const BTOperator                                    &BT_operator);

        void vmult(VectorType       &dst,
                   const VectorType &src) const;

      private:
        const AInvOperator                                  &A_inverse_operator;
        const SchurComplementOperator<PressureVectorType>   &S_inverse_operator;
        const BTOperator                                    &BT_operator;
        mutable VectorType                                   tmp;
    };

    template <class AInvOperator, class BTOperator, class VectorType, class PressureVectorType>
    BlockSchurPreconditioner<AInvOperator, BTOperator, VectorType, PressureVectorType>::
    BlockSchurPreconditioner(
      const AInvOperator                                  &A_inverse_operator,
      const SchurComplementOperator<PressureVectorType>   &S_inverse_operator,
      const BTOperator                                    &BT_operator)
      :
      A_inverse_operator(A_inverse_operator),
      S_inverse_operator(S_inverse_operator),
      BT_operator(BT_operator)
    {}





    template <class AInvOperator, class BTOperator, class VectorType, class PressureVectorType>
    void
    BlockSchurPreconditioner<AInvOperator, BTOperator, VectorType, PressureVectorType>::
    vmult(VectorType       &dst,
          const VectorType &src) const
    {
      if (tmp.size() == 0)
        tmp.reinit(src);

      dst = 0.0;

      {
        S_inverse_operator.vmult(dst.block(1), src.block(1));
        dst.block(1) *= -1.0;
      }

      {
        if constexpr (std::is_same_v<VectorType, dealii::LinearAlgebra::distributed::BlockVector<double>>)
          BT_operator.vmult(tmp, dst);
        else
          BT_operator.vmult(tmp.block(0), dst.block(1));

        tmp.block(0) *= -1.0;
        tmp.block(0) += src.block(0);
      }

      A_inverse_operator.vmult(dst.block(0), tmp.block(0));
    }

    template<class OperatorType, class StokesMatrixType, class SchurComplementMatrixType, class VectorType>
    class SchurApproximation : public SchurComplementOperator<VectorType>
    {
      public:
        SchurApproximation(const OperatorType &schur_preconditioner,
                           const StokesMatrixType &stokes_matrix,
                           const SchurComplementMatrixType &Schur_complement_block,
                           const bool do_solve_Schur_complement,
                           const double Schur_complement_tolerance);

        void vmult(VectorType &dst, const VectorType &src) const override;
        unsigned int n_iterations() const override;

      private:
        const OperatorType &schur_preconditioner;
        const StokesMatrixType &stokes_matrix;
        const SchurComplementMatrixType &Schur_complement_block;
        const bool do_solve_Schur_complement;
        const double Schur_complement_tolerance;
        mutable unsigned int n_iterations_Schur_complement_;
    };



    template <class OperatorType, class StokesMatrixType, class SchurComplementMatrixType, class VectorType>
    SchurApproximation<OperatorType, StokesMatrixType, SchurComplementMatrixType, VectorType>::SchurApproximation(const OperatorType &schur_preconditioner,
        const StokesMatrixType &stokes_matrix,
        const SchurComplementMatrixType &Schur_complement_block,
        const bool do_solve_Schur_complement,
        const double Schur_complement_tolerance)
      :
      schur_preconditioner(schur_preconditioner),
      stokes_matrix(stokes_matrix),
      Schur_complement_block(Schur_complement_block),
      do_solve_Schur_complement(do_solve_Schur_complement),
      Schur_complement_tolerance(Schur_complement_tolerance),
      n_iterations_Schur_complement_(0)
    {}







    template<class OperatorType, class StokesMatrixType, class SchurComplementMatrixType, class VectorType>
    void SchurApproximation<OperatorType, StokesMatrixType, SchurComplementMatrixType, VectorType>::vmult(VectorType &dst,
        const VectorType &src) const
    {
      if (do_solve_Schur_complement)
        {

          SolverControl solver_control(100, src.l2_norm() * Schur_complement_tolerance,true);

          SolverCG<VectorType> solver(solver_control);
          // Trilinos reports a breakdown in case src=dst=0, even
          // though it should return convergence without
          // iterating. We simply skip solving in this case.
          if (src.l2_norm() > 1e-50)
            {
              try
                {
                  // explicitly zero out because GMRES does not guarantee that dst is zeroed out
                  dst = 0.0;

                  solver.solve(Schur_complement_block,
                               dst, src,
                               schur_preconditioner);
                  n_iterations_Schur_complement_ += solver_control.last_step();
                }
              // if the solver fails, report the error from processor 0 with some additional
              // information about its location, and throw a quiet exception on all other
              // processors
              catch (const std::exception &exc)
                {
                  Utilities::throw_linear_solver_failure_exception("iterative (bottom right) solver",
                                                                   "BlockSchurPreconditioner::vmult",
                                                                   std::vector<SolverControl> {solver_control},
                                                                   exc,
                                                                   src.get_mpi_communicator());
                }
            }
        }
      else
        {
          dst = 0.0;
          schur_preconditioner.vmult(dst,src);
          n_iterations_Schur_complement_ += 1;
        }
    }



    template<class OperatorType, class StokesMatrixType, class SchurComplementMatrixType, class VectorType>
    unsigned int SchurApproximation<OperatorType, StokesMatrixType, SchurComplementMatrixType, VectorType>::n_iterations() const
    {
      return n_iterations_Schur_complement_;
    }






    template<class StokesMatrixType, class BOperatorType, class BTOperatorType>
    class BC_invBT_Operator
    {
      public:
        BC_invBT_Operator(const StokesMatrixType &system_matrix,
                          const BOperatorType &B_operator,
                          const BTOperatorType &BT_operator,
                          const dealii::LinearAlgebra::distributed::Vector<double> &diag_A_inv):
          system_matrix(system_matrix),
          B_operator(B_operator),
          BT_operator(BT_operator),
          diag_A_inv(diag_A_inv)
        {}
        void vmult(dealii::LinearAlgebra::distributed::Vector<double> &dst,
                   const dealii::LinearAlgebra::distributed::Vector<double> &src) const;
      private:
        const StokesMatrixType &system_matrix;
        const BOperatorType &B_operator;
        const BTOperatorType &BT_operator;
        const dealii::LinearAlgebra::distributed::Vector<double> &diag_A_inv;
    };

    template <class StokesMatrixType, class AOperatorType, class BOperatorType, class BTOperatorType, class SchurComplementMatrixType,class VectorType, class PreconditionerMp>
    class DiagBFBT: public SchurComplementOperator<VectorType>
    {
      public:
        /**
         * Constructor.
         * @param mp_preconditioner The preconditioner for the BC^{-1}B^T operator
         *        used in the inner CG solves.
         * @param do_solve_schur_complement Full solve with Schur complement or just vmult.
         * @param solver_tolerance The relative solver tolerance for the inner CG solves
         *        with BC^{-1}B^T.
         * @param diag_A_inv Diagonal of A used as C and D in the diag A-BFBT preconditioner.
         * @param system_matrix The Stokes operator of the form
         * [A B^T
         *  B 0].
         * @param A_operator The velocity block operator.
         * @param B_operator The B block operator.
         * @param BT_operator The B^T block operator.
         * @param mp_matrix Pressure mass matrix used as preconditioner for BC^{-1}B^T.
         */
        DiagBFBT(const PreconditionerMp &mp_preconditioner,
                 const bool do_solve_schur_complement,
                 const double solver_tolerance,
                 const dealii::LinearAlgebra::distributed::Vector<double> &diag_A_inv,
                 const StokesMatrixType &system_matrix,
                 const AOperatorType &A_operator,
                 const BOperatorType &B_operator,
                 const BTOperatorType &BT_operator,
                 const SchurComplementMatrixType &mp_matrix);

        void vmult(VectorType &dst,
                   const VectorType &src) const override;

        unsigned int n_iterations() const override;

      private:
        mutable unsigned int n_iterations_;
        const PreconditionerMp &mp_preconditioner;
        const bool do_solve_schur_complement;
        const double solver_tolerance;
        const dealii::LinearAlgebra::distributed::Vector<double> &diag_A_inv;
        const StokesMatrixType &system_matrix;
        const AOperatorType &A_operator;
        const BOperatorType &B_operator;
        const BTOperatorType &BT_operator;
        const SchurComplementMatrixType &mp_matrix;
    };



  }
}



#endif
