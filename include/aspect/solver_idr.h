// ---------------------------------------------------------------------
//
// Copyright (C) 2000 - 2019 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_solver_idr_h
#define dealii_solver_idr_h


#include <deal.II/base/config.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>

#include <cmath>

DEAL_II_NAMESPACE_OPEN

/*!@addtogroup Solvers */
/*@{*/

/**
 * IDR(s)
 *
 *
 * @author Conrad Clevenger, 2019
 */
template <class VectorType = Vector<double>>
class SolverIDR : public SolverBase<VectorType>
{
  public:
    /**
     * Structure for storing additional data needed by the solver. Here
     * it only stores the order s of the IDR(s) method. By default we
     * set s equal to 2.
     */
    struct AdditionalData
    {
      explicit AdditionalData(const unsigned int s = 2)
        : s(s)
      {}

      const unsigned int s;
    };

    /**
     * Constructor.
     */
    SolverIDR(SolverControl            &cn,
              VectorMemory<VectorType> &mem,
              const AdditionalData     &data = AdditionalData(2));

    /**
     * Constructor. Use an object of type GrowingVectorMemory as a default to
     * allocate memory.
     */
    SolverIDR(SolverControl &cn, const AdditionalData &data = AdditionalData(2));

    /**
     * Virtual destructor.
     */
    virtual ~SolverIDR() override = default;

    /**
     * Solve the linear system $Ax=b$ for x.
     */
    template <typename MatrixType, typename PreconditionerType>
    void
    solve(const MatrixType         &A,
          VectorType               &x,
          const VectorType         &b,
          const PreconditionerType &preconditioner);

  protected:
    /**
     * Interface for derived class. This function gets the current iteration
     * vector, the residual and the update vector in each step. It can be used
     * for graphical output of the convergence history.
     */
    virtual void
    print_vectors(const unsigned int step,
                  const VectorType &x,
                  const VectorType &r,
                  const VectorType &d) const;

  private:
    /**
     * Set of s random orthonormalized vectors.
     */
    std::vector<VectorType> Q;

    /**
     * Additional solver parameters.
     */
    AdditionalData additional_data;
};

/*@}*/
/*------------------------- Implementation ----------------------------*/

#ifndef DOXYGEN

template <class VectorType>
SolverIDR<VectorType>::SolverIDR(SolverControl            &cn,
                                 VectorMemory<VectorType> &mem,
                                 const AdditionalData     &data)
  : SolverBase<VectorType>(cn, mem)
  , additional_data(data)

{}



template <class VectorType>
SolverIDR<VectorType>::SolverIDR(SolverControl &cn, const AdditionalData &data)
  : SolverBase<VectorType>(cn)
  , additional_data(data)

{}



template <class VectorType>
void
SolverIDR<VectorType>::print_vectors(const unsigned int,
                                     const VectorType &,
                                     const VectorType &,
                                     const VectorType &) const
{}



template <class VectorType>
template <typename MatrixType, typename PreconditionerType>
void
SolverIDR<VectorType>::solve(const MatrixType         &A,
                             VectorType               &x,
                             const VectorType         &b,
                             const PreconditionerType &preconditioner)
{
  SolverControl::State iteration_state = SolverControl::iterate;
  unsigned int         step            = 0;

  const unsigned int s = additional_data.s;

  // Initial residual
  VectorType r;
  r.reinit(x, true);
  A.vmult(r, x);
  r.sadd(-1.0, 1.0, b);

  // Check for really good initial guess...
  double res      = r.l2_norm();
  iteration_state = this->iteration_status(step, res, x);
  if (iteration_state == SolverControl::success)
    return;

  // Initialization
  VectorType v, vhat, uhat, ghat;
  v.reinit(x, true);
  vhat.reinit(x, true);
  uhat.reinit(x, true);
  ghat.reinit(x, true);

  std::vector<VectorType> G(s);
  std::vector<VectorType> U(s);
  FullMatrix<double>      M(s, s);
  for (unsigned int i = 0; i < s; ++i)
    {
      G[i].reinit(x, true);
      U[i].reinit(x, true);
      G[i] = 0;
      U[i] = 0;

      M(i, i) = 1.;
    }
  double omega = 1.;

  // Compute random set of s orthonormalized vectors Q
  {
    Q.resize(s);
    for (unsigned int i = 0; i < s; ++i)
      {
        Q[i].reinit(x, true);
        for (auto indx : Q[i].locally_owned_elements())
          Q[i](indx) = Utilities::generate_normal_random_number(0.0, 1.0);
        Q[i].compress(VectorOperation::insert);

        for (unsigned int j = 0; j < i; ++j)
          {
            v = Q[j];
            v *= (Q[j] * Q[i]) / (Q[i] * Q[i]);
            Q[i].add(-1.0, v);
          }
        Q[i] *= 1.0 / Q[i].l2_norm();
      }
  }

  bool early_exit = false;
  while (iteration_state == SolverControl::iterate)
    {
      ++step;

      // Compute phi
      Vector<double> phi(s);
      for (unsigned int i = 0; i < s; ++i)
        phi(i) = Q[i] * r;

      for (unsigned int k = 0; k < s; ++k)
        {
          // M(k:s)*gamma = phi(k:s)
          Vector<double> gamma(s - k);
          {
            Vector<double>            phik(s - k);
            FullMatrix<double>        Mk(s - k, s - k);
            std::vector<unsigned int> indices;
            unsigned int              j = 0;
            for (unsigned int i = k; i < s; ++i)
              {
                indices.push_back(i);
                phik(j) = phi(i);
                ++j;
              }
            Mk.extract_submatrix_from(M, indices, indices);

            Mk.gauss_jordan();
            Mk.vmult(gamma, phik);
          }

          {
            v = r;

            unsigned int j = 0;
            for (unsigned int i = k; i < s; ++i)
              {
                v.add(-1.0 * gamma(j), G[i]);
                ++j;
              }
            preconditioner.vmult(vhat, v);

            uhat = vhat;
            uhat *= omega;
            j = 0;
            for (unsigned int i = k; i < s; ++i)
              {
                uhat.add(gamma(j), U[i]);
                ++j;
              }
            A.vmult(ghat, uhat);
          }

          // Update G and U
          // Orthogonalize ghat to Q0,..,Q_{k-1}
          // and update uhat
          for (unsigned int i = 0; i < k; ++i)
            {
              double alpha = (Q[i] * ghat) / M(i, i);
              ghat.add(-1.0 * alpha, G[i]);
              uhat.add(-1.0 * alpha, U[i]);
            }
          G[k] = ghat;
          U[k] = uhat;

          // Update kth column of M
          for (unsigned int i = k; i < s; ++i)
            M(i, k) = Q[i] * G[k];

          // Orthoginalize r to Q0,...,Qk,
          // update x
          {
            double beta = phi(k) / M(k, k);
            r.add(-1.0 * beta, G[k]);
            x.add(beta, U[k]);

            // Check for zero residual to avoid breakdown later
            res = r.l2_norm();
            if (res < 1e-14)
              {
                early_exit = true;
                iteration_state = SolverControl::success;
                break;
              }

            // Update phi
            if (k + 1 < s)
              {
                for (unsigned int i = 0; i < k + 1; ++i)
                  phi(i) = 0.0;
                for (unsigned int i = k + 1; i < s; ++i)
                  phi(i) -= beta * M(i, k);
              }
          }
        }
      if (early_exit == true)
        {
          res             = r.l2_norm();
          iteration_state = this->iteration_status(step, res, x);
          break;
        }

      // Update r and x
      preconditioner.vmult(vhat, r);
      A.vmult(v, vhat);

      omega = (v * r) / (v * v);

      r.add(-1.0 * omega, v);
      x.add(omega, vhat);

      // Check for convergence
      res             = r.l2_norm();
      iteration_state = this->iteration_status(step, res, x);
      if (iteration_state != SolverControl::iterate)
        break;
    }

  if (iteration_state != SolverControl::success)
    AssertThrow(false, SolverControl::NoConvergence(step, res));
}


#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
