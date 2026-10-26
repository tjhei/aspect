/*
  Copyright (C) 2018 - 2026 by the authors of the ASPECT code.

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

#include <algorithm>

#include <aspect/simulator/solver/stokes_matrix_free.h>
#include <aspect/simulator/solver/matrix_free_operators.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/melt.h>
#include <aspect/newton.h>
#include <aspect/utilities.h>

#include <boost/geometry/strategies/strategy_transform.hpp>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/signaling_nan.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_evaluation_data.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/read_write_vector.templates.h>

#include <deal.II/grid/manifold.h>

#include <deal.II/matrix_free/tools.h>
#include <vector>

namespace aspect
{
  namespace internal
  {

    template<typename VectorType, typename SmootherType>
    void MGSmootherRemoveNullspace<VectorType, SmootherType>::clear()
    {}

    template<typename VectorType, typename SmootherType>
    void MGSmootherRemoveNullspace<VectorType, SmootherType>::smooth(const unsigned int level,
                                                                     VectorType &dst, const VectorType &src) const
    {
      smoother->smooth(level, dst, src);
      dst.add(-dst.mean_value());
    }


    template<typename VectorType, typename SmootherType>
    void MGSmootherRemoveNullspace<VectorType, SmootherType>::apply(const unsigned int level, VectorType &dst,
                                                                    const VectorType &src) const
    {
      smoother->apply(level,dst,src);
      dst.add(-dst.mean_value());
    }

    template<typename VectorType, typename SmootherType>
    void MGSmootherRemoveNullspace<VectorType, SmootherType>::initialize(const SmootherType &smoother)
    {
      this->smoother=&smoother;
    }
    template<typename VectorType>

    void MGCoarseGridApplySmootherRemoveNullspace<VectorType>::operator()(const unsigned int level,
                                                                          VectorType &dst,
                                                                          const VectorType &src) const
    {
      (*coarse_grid_solver)(level,dst,src);
      dst.add(-dst.mean_value());
    }
    template<typename VectorType>

    void MGCoarseGridApplySmootherRemoveNullspace<VectorType>::initialize(const
                                                                          dealii::MGCoarseGridApplySmoother<VectorType> &coarse_grid_solver)
    {
      this->coarse_grid_solver = &coarse_grid_solver;
    }
    template class MGCoarseGridApplySmootherRemoveNullspace<dealii::LinearAlgebra::distributed::Vector<double>>;


    /**
     * Matrix-free operators must use deal.II defined vectors, while the rest of the ASPECT
     * software is based on Trilinos vectors. Here we define functions which copy between the
     * vector types.
     */
    namespace ChangeVectorTypes
    {
      void copy(aspect::LinearAlgebra::Vector &out,
                const dealii::LinearAlgebra::distributed::Vector<double> &in)
      {
        dealii::LinearAlgebra::ReadWriteVector<double> rwv(out.locally_owned_elements());
        rwv.import_elements(in, VectorOperation::insert);
        out.import_elements(rwv,VectorOperation::insert);
      }

      void copy(dealii::LinearAlgebra::distributed::Vector<double> &out,
                const aspect::LinearAlgebra::Vector &in)
      {
        dealii::LinearAlgebra::ReadWriteVector<double> rwv;

#ifndef ASPECT_USE_TPETRA
        rwv.reinit(in);
#else
        rwv.reinit(in.locally_owned_elements());
        rwv.import_elements(in, VectorOperation::insert);
#endif

        out.import_elements(rwv, VectorOperation::insert);
      }

      void copy(aspect::LinearAlgebra::BlockVector &out,
                const dealii::LinearAlgebra::distributed::BlockVector<double> &in)
      {
        const unsigned int n_blocks = in.n_blocks();
        for (unsigned int b=0; b<n_blocks; ++b)
          copy(out.block(b),in.block(b));
      }

      void copy(dealii::LinearAlgebra::distributed::BlockVector<double> &out,
                const aspect::LinearAlgebra::BlockVector &in)
      {
        const unsigned int n_blocks = in.n_blocks();
        for (unsigned int b=0; b<n_blocks; ++b)
          copy(out.block(b),in.block(b));
      }
    }

  }


  namespace MatrixFreeStokesOperators
  {
    template <int dim, typename number>
    inline std::size_t
    OperatorCellData<dim,number>::memory_consumption() const
    {
      return viscosity.memory_consumption()
             + newton_factor_wrt_pressure_table.memory_consumption()
             + strain_rate_table.memory_consumption()
             + newton_factor_wrt_strain_rate_table.memory_consumption()
             + dilation_derivative_wrt_pressure_table.memory_consumption()
             + dilation_derivative_wrt_strain_rate_table.memory_consumption();
    }



    template <int dim, typename number>
    void
    OperatorCellData<dim,number>::clear()
    {
      enable_newton_derivatives = false;
      viscosity.clear();
      newton_factor_wrt_pressure_table.clear();
      strain_rate_table.clear();
      newton_factor_wrt_strain_rate_table.clear();
      dilation_derivative_wrt_pressure_table.clear();
      dilation_derivative_wrt_strain_rate_table.clear();
    }



    template <int dim, typename number>
    void
    fill_active_cell_data(const DoFHandler<dim> &dof_handler,
                          const DoFHandler<dim> &dof_handler_projection,
                          const Introspection<dim> &introspection,
                          const Quadrature<dim> &quadrature_formula,
                          const MaterialModel::Interface<dim> &material_model,
                          const MaterialModel::MaterialAveraging::AveragingOperation &material_averaging,
                          const Mapping<dim> &mapping,
                          const MatrixFree<dim, double> &matrix_free,
                          const MatrixFree<dim, double> &matrix_free_schur,
                          const MatrixFree<dim, double> &matrix_free_A,
                          const LinearAlgebra::BlockVector &current_linearization_point,
                          const MPI_Comm &mpi_comm,
                          dealii::LinearAlgebra::distributed::Vector<double> &active_viscosity_vector,
                          OperatorCellData<dim, number> &active_cell_data,
                          double &minimum_viscosity,
                          double &maximum_viscosity)
    {
#ifndef DEBUG
      (void)matrix_free_schur;
      (void)matrix_free_A;
#endif

      double minimum_viscosity_local = std::numeric_limits<double>::max();
      double maximum_viscosity_local = std::numeric_limits<double>::lowest();

      const bool active_no_averaging = (material_averaging
                                        ==
                                        MaterialModel::MaterialAveraging::AveragingOperation::none);

      {
        // allocate space for viscosities

        const unsigned int n_cells = matrix_free.n_cell_batches();
        const unsigned int n_q_points = quadrature_formula.size();

        // One value per cell is required for DGQ0 projection and n_q_points
        // values per cell for DGQ1.
        if (active_no_averaging || dof_handler_projection.get_fe().degree == 1)
          {
            active_cell_data.viscosity.reinit(TableIndices<2>(n_cells, n_q_points));
          }
        else if (dof_handler_projection.get_fe().degree == 0)
          active_cell_data.viscosity.reinit(TableIndices<2>(n_cells, 1));
        else
          Assert(false, ExcInternalError());
      }

      // Fill the DGQ0 or DGQ1 vector of viscosity values on the active mesh
      {
        FEValues<dim> fe_values (mapping,
                                 dof_handler.get_fe(),
                                 quadrature_formula,
                                 update_values   |
                                 update_gradients |
                                 update_quadrature_points |
                                 update_JxW_values);

        MaterialModel::MaterialModelInputs<dim> in(fe_values.n_quadrature_points, introspection.n_compositional_fields);
        in.requested_properties = MaterialModel::MaterialProperties::viscosity;
        MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points, introspection.n_compositional_fields);

        // This function call computes a cellwise projection of data defined at quadrature points to
        // a vector defined by the projection DoFHandler. As an input, we must define a lambda which returns
        // a viscosity value for each quadrature point of the given cell. The projection is then stored in
        // the active level viscosity vector provided.
        Utilities::project_cellwise(mapping,
                                    dof_handler_projection,
                                    0,
                                    quadrature_formula,
                                    [&](const typename DoFHandler<dim>::active_cell_iterator & cell,
                                        const std::vector<Point<dim>> & /*q_points*/,
                                        std::vector<double> &values) -> void
        {
          const typename DoFHandler<dim>::active_cell_iterator FEQ_cell(&dof_handler.get_triangulation(),
          cell->level(),
          cell->index(),
          &dof_handler);

          fe_values.reinit (FEQ_cell);
          in.reinit(fe_values, FEQ_cell, introspection, current_linearization_point);

          // Query the material model for the active level viscosities
          material_model.fill_additional_material_model_inputs(in, current_linearization_point, fe_values, introspection);
          material_model.evaluate(in, out);

          // If using a cellwise average for viscosity, average the values here.
          // When the projection is computed, this will set the viscosity exactly
          // to this averaged value.
          if (dof_handler_projection.get_fe().degree == 0)
            MaterialModel::MaterialAveraging::average (material_averaging,
            FEQ_cell,
            quadrature_formula,
            mapping,
            in.requested_properties,
            out);

          if (active_no_averaging)
            {
              // also copy the viscosity to the CellData

              const unsigned int index = matrix_free.get_matrix_free_cell_index(cell);
              const unsigned int lane_size = VectorizedArray<number>::size();
              const unsigned int cell_index = index / lane_size;
              const unsigned int lane = index % lane_size;

              for (unsigned int q=0; q<values.size(); ++q)
                active_cell_data.viscosity(cell_index, q)[lane] = out.viscosities[q];
            }

          for (unsigned int i=0; i<values.size(); ++i)
            {
              // Find the local max/min of the evaluated viscosities.
              minimum_viscosity_local = std::min(minimum_viscosity_local, out.viscosities[i]);
              maximum_viscosity_local = std::max(maximum_viscosity_local, out.viscosities[i]);

              values[i] = out.viscosities[i];
            }
        },
        active_viscosity_vector);
        active_viscosity_vector.compress(VectorOperation::insert);
      }

      minimum_viscosity = dealii::Utilities::MPI::min(minimum_viscosity_local, mpi_comm);
      maximum_viscosity = dealii::Utilities::MPI::max(maximum_viscosity_local, mpi_comm);

      FEValues<dim> fe_values_projection (mapping,
                                          dof_handler_projection.get_fe(),
                                          quadrature_formula,
                                          update_values);

      // Create active mesh viscosity table.
      {
        const unsigned int n_cells = matrix_free.n_cell_batches();
        const unsigned int n_q_points = quadrature_formula.size();
        std::vector<double> values_on_quad (n_q_points);

        std::vector<types::global_dof_index> local_dof_indices(dof_handler_projection.get_fe().dofs_per_cell);
        for (unsigned int cell=0; cell<n_cells; ++cell)
          {
            const unsigned int n_components_filled = matrix_free.n_active_entries_per_cell_batch(cell);

            for (unsigned int i=0; i<n_components_filled; ++i)
              {
                typename DoFHandler<dim>::active_cell_iterator FEQ_cell =
                  matrix_free.get_cell_iterator(cell,i);
                typename DoFHandler<dim>::active_cell_iterator DG_cell(&dof_handler_projection.get_triangulation(),
                                                                       FEQ_cell->level(),
                                                                       FEQ_cell->index(),
                                                                       &dof_handler_projection);
                DG_cell->get_active_or_mg_dof_indices(local_dof_indices);

#ifdef DEBUG
                {
                  // Verify that all MatrixFree objects iterate over cells in the same way:
                  typename DoFHandler<dim>::active_cell_iterator s_cell =
                    matrix_free_schur.get_cell_iterator(cell,i,1);
                  double distance_s = s_cell->center().distance(FEQ_cell->center());
                  Assert(distance_s < 1e-10, ExcInternalError());

                  typename DoFHandler<dim>::active_cell_iterator A_cell =
                    matrix_free_A.get_cell_iterator(cell,i);
                  double distance_A = A_cell->center().distance(FEQ_cell->center());
                  Assert(distance_A < 1e-10, ExcInternalError());
                }
#endif

                // For DGQ0, we simply use the viscosity at the single
                // support point of the element. For DGQ1, we must project
                // back to quadrature point values.
                if (active_no_averaging)
                  {
                    // do nothing, filled above already!
                  }
                else if (dof_handler_projection.get_fe().degree == 0)
                  active_cell_data.viscosity(cell, 0)[i] = active_viscosity_vector(local_dof_indices[0]);
                else
                  {
                    fe_values_projection.reinit(DG_cell);
                    fe_values_projection.get_function_values(active_viscosity_vector,
                                                             local_dof_indices,
                                                             values_on_quad);

                    // Do not allow viscosity to be greater than or less than the limits
                    // of the evaluated viscosity on the active level.
                    for (unsigned int q=0; q<n_q_points; ++q)
                      active_cell_data.viscosity(cell, q)[i]
                        = std::clamp(values_on_quad[q], minimum_viscosity, maximum_viscosity);
                  }
              }
          }
      }
    }
  }

  //class to lump velocity mass matrix for weighted BFBT by Rudi et al (2017).
  template<int dim,int degree_v, typename number>
  MatrixFreeStokesOperators::LumpedVelocityMassOperator<dim,degree_v,number>::LumpedVelocityMassOperator():
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>(),
    cell_data(nullptr)
  {}

  template<int dim, int degree_v, typename number>
  void MatrixFreeStokesOperators::LumpedVelocityMassOperator<dim, degree_v, number>::set_cell_data(const OperatorCellData<dim, number> &data)
  {
    this->cell_data=&data;
  }

  template<int dim, int degree_v, typename number>
  void MatrixFreeStokesOperators::LumpedVelocityMassOperator<dim, degree_v, number>
  ::local_apply( const dealii::MatrixFree<dim,number> &data,
                 dealii::LinearAlgebra::distributed::Vector<number> &dst,
                 const dealii::LinearAlgebra::distributed::Vector<number> &src,
                 const ::std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim,  degree_v,  degree_v+1, dim,  number> velocity(data,0);
    const bool constant_per_cell(cell_data->viscosity.size(1)==1);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit(cell);
        velocity.gather_evaluate(src,EvaluationFlags::values);
        const unsigned int n_filled=data.n_active_entries_per_cell_batch(cell);
        for (const unsigned int q: velocity.quadrature_point_indices())
          {
            VectorizedArray<number> w=0.;
            for (unsigned int c=0; c<n_filled; ++c)
              w[c]=std::sqrt(constant_per_cell?cell_data->viscosity(cell,0)[c]:
                             cell_data->viscosity(cell,q)[c]);
            velocity.submit_value(w*velocity.get_value(q),q);
          }
        velocity.integrate_scatter(EvaluationFlags::values,dst);
      }
  }

  template<int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::LumpedVelocityMassOperator<dim, degree_v,  number>
  ::apply_add(dealii::LinearAlgebra::distributed::Vector<number> &dst, const dealii::LinearAlgebra::distributed::Vector<number> &src) const
  {
    MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number>>::
    data->cell_loop(&LumpedVelocityMassOperator::local_apply, this, dst,src);
  }

  template<int dim, int degree_v, typename number>
  void MatrixFreeStokesOperators::LumpedVelocityMassOperator<dim, degree_v,  number>::compute_diagonal()
  {
    Assert(this->cell_data!=nullptr,ExcNotInitialized());
    this->inverse_diagonal_entries=std::make_shared<DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>>();
    dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal=
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    dealii::LinearAlgebra::distributed::Vector<number> row_sums;
    this->data->initialize_dof_vector(row_sums);
    inverse_diagonal=number(1.0);
    this->vmult(row_sums,inverse_diagonal);
    this->set_constrained_entries_to_one(row_sums);

    for (unsigned int i=0; i<inverse_diagonal.locally_owned_size(); ++i)
      {
        Assert(row_sums.local_element(i)>0.,
               ExcMessage("Lumped velocity mass matrix entries must be positive."));
        inverse_diagonal.local_element(i)=1./row_sums.local_element(i);
      }
    inverse_diagonal.update_ghost_values();

  }

  template <int dim, int degree_v, typename number>
  MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>::StokesOperator ()
    :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number>>()
  {}



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>::clear ()
  {
    this->cell_data = nullptr;
    MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::BlockVector<number>>::clear();
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>::
  set_cell_data (const OperatorCellData<dim,number> &data)
  {
    this->cell_data = &data;
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>
  ::compute_diagonal ()
  {
    // There is currently no need in the code for the diagonal of the entire Stokes
    // block. If needed, one could easily construct based on the diagonal of the A
    // block and append zeros to the end for the number of pressure DoFs.
    Assert(false, ExcNotImplemented());
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>
  ::local_apply (const dealii::MatrixFree<dim, number>                         &data,
                 dealii::LinearAlgebra::distributed::BlockVector<number>       &dst,
                 const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
                 const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim,degree_v,degree_v+1,dim,number> u_eval(data, 0);
    FEEvaluation<dim,degree_v-1,degree_v+1,1,number> p_eval(data, /*dofh*/1);

    const bool use_viscosity_at_quadrature_points
      = (cell_data->viscosity.size(1) == u_eval.n_q_points);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        VectorizedArray<number> viscosity_x_2 = 2. * cell_data->viscosity(cell,0);

        u_eval.reinit(cell);
        u_eval.gather_evaluate(src.block(0), EvaluationFlags::gradients);

        p_eval.reinit(cell);
        p_eval.gather_evaluate(src.block(1), EvaluationFlags::values);

        // Derivative terms related to the Newton solver
        VectorizedArray<number> deta_deps_times_sym_grad_u(0.);
        VectorizedArray<number> deta_dp_times_p(0.);
        VectorizedArray<number> eps_times_sym_grad_u(0.);
        VectorizedArray<number> eps_times_sym_grad_u_JxW(0.);
        VectorizedArray<number> theta_times_div_u(0.);
        VectorizedArray<number> theta_times_div_u_JxW(0.);

        // Pre-compute the averaged values
        if (cell_data->enable_newton_derivatives
            && cell_data->average_newton_factors)
          {
            SymmetricTensor<2,dim,VectorizedArray<number>> sym_grad_u;
            VectorizedArray<number> val_p;
            for (const unsigned int q : u_eval.quadrature_point_indices())
              {
                sym_grad_u = u_eval.get_symmetric_gradient(q);
                val_p      = p_eval.get_value(q);
                deta_deps_times_sym_grad_u += cell_data->newton_factor_wrt_strain_rate_table(cell,q)
                                              * sym_grad_u;
                deta_dp_times_p += cell_data->newton_factor_wrt_pressure_table(cell,q) * val_p;
                if (cell_data->symmetrize_newton_system)
                  {
                    // The symmetrization is more complicated than it looks:
                    // The averaged Newton term is expressed as
                    // 2 (\sum_q JxW_q symgrad_phi_Iq : epsilon_q) (\sum_r w_r deta_depsilon_r : symgrad_phi_Jr)
                    // when symmetrized, it becomes
                    // (\sum_q JxW_q symgrad_phi_Iq : epsilon_q) (\sum_r w_r deta_depsilon_r : symgrad_phi_Jr) +
                    // (\sum_r w_r symgrad_phi_Ir : deta_depsilon_r) (\sum_q JxW_q epsilon_q : symgrad_phi_Jq)
                    // Let us focus on the second term. In the matrix-free framework, we should submit a
                    // symmetric tensor for each quadrature point q to be multiplied by the test function
                    // symgrad_phi_Iq and the quadrature weight JxW_q. In order to do so, we first rewrite the
                    // second term by swapping q and r:
                    // \sum_q\sum_r JxW_r w_q (symgrad_phi_Iq : deta_depsilon_q) (epsilon_r : symgrad_phi_Jr)
                    // In the above expression, the test function is separated out, but the quadrature weight
                    // is not. To separate JxW_q, we further rewrite the second term by
                    // \sum_q\sum_r JxW_q (JxW_r / JxW_q) w_q (symgrad_phi_Iq : deta_depsilon_q) (epsilon_r : symgrad_phi_Jr)
                    // Thus, the term to be submitted for each quadrature point q is
                    // \sum_r (JxW_r / JxW_q) w_q deta_depsilon_q (epsilon_r : symgrad_phi_Jr)
                    // That is the reason why we need the term (epsilon_r : symgrad_phi_Jr) JxW_r.
                    eps_times_sym_grad_u_JxW += cell_data->strain_rate_table(cell,q) * sym_grad_u * u_eval.JxW(q);
                    // The same goes for the term corresponding to compressibility/dilation
                    if (cell_data->is_compressible || cell_data->enable_prescribed_dilation)
                      theta_times_div_u_JxW += trace(cell_data->strain_rate_table(cell,q)) * trace(sym_grad_u) * u_eval.JxW(q);
                  }
              }
          }

        for (const unsigned int q : u_eval.quadrature_point_indices())
          {
            // Only update the viscosity if a Q1 projection is used.
            if (use_viscosity_at_quadrature_points)
              viscosity_x_2 = 2. * cell_data->viscosity(cell,q);

            const SymmetricTensor<2,dim,VectorizedArray<number>>
            sym_grad_u = u_eval.get_symmetric_gradient(q);
            const VectorizedArray<number> div_u = trace(sym_grad_u);
            const VectorizedArray<number> val_p = p_eval.get_value(q);

            // Terms to be tested by phi_p:
            VectorizedArray<number> pressure_terms =
              -cell_data->pressure_scaling * div_u;

            if (cell_data->enable_prescribed_dilation)
              pressure_terms -= cell_data->pressure_scaling *
                                cell_data->pressure_scaling *
                                cell_data->dilation_lhs_term_table(cell,q) *
                                val_p;

            // Terms to be tested by the symmetric gradients of phi_u:
            SymmetricTensor<2,dim,VectorizedArray<number>>
            velocity_terms = viscosity_x_2 * sym_grad_u;

            for (unsigned int d=0; d<dim; ++d)
              velocity_terms[d][d] -= cell_data->pressure_scaling * val_p;

            if (cell_data->is_compressible ||
                cell_data->enable_prescribed_dilation)
              for (unsigned int d=0; d<dim; ++d)
                velocity_terms[d][d] -= viscosity_x_2 / 3. * div_u;

            // Add the Newton derivatives if required.
            if (cell_data->enable_newton_derivatives)
              {
                if (cell_data->average_newton_factors)
                  {
                    if (cell_data->symmetrize_newton_system)
                      eps_times_sym_grad_u = eps_times_sym_grad_u_JxW / u_eval.JxW(q);
                  }
                else
                  {
                    deta_deps_times_sym_grad_u = cell_data->newton_factor_wrt_strain_rate_table(cell,q)
                                                 * sym_grad_u;
                    deta_dp_times_p = cell_data->newton_factor_wrt_pressure_table(cell,q) * val_p;
                    if (cell_data->symmetrize_newton_system)
                      eps_times_sym_grad_u = cell_data->strain_rate_table(cell,q) * sym_grad_u;
                  }

                velocity_terms +=
                  ( cell_data->symmetrize_newton_system ?
                    ( cell_data->strain_rate_table(cell,q) * deta_deps_times_sym_grad_u +
                      cell_data->newton_factor_wrt_strain_rate_table(cell,q) * eps_times_sym_grad_u ) :
                    2. * cell_data->strain_rate_table(cell,q) * deta_deps_times_sym_grad_u )
                  +
                  2. * cell_data->strain_rate_table(cell,q) * deta_dp_times_p;

                if (cell_data->is_compressible ||
                    cell_data->enable_prescribed_dilation)
                  {
                    constexpr number one_third = 1.0 / 3.0;
                    if (cell_data->symmetrize_newton_system)
                      {
                        if (cell_data->average_newton_factors)
                          theta_times_div_u = theta_times_div_u_JxW / u_eval.JxW(q);
                        else
                          theta_times_div_u = trace(cell_data->strain_rate_table(cell,q)) * div_u;

                        velocity_terms -= one_third * cell_data->newton_factor_wrt_strain_rate_table(cell,q) * theta_times_div_u;
                        for (unsigned int d = 0; d < dim; ++d)
                          velocity_terms[d][d] -= one_third * trace(cell_data->strain_rate_table(cell,q)) * deta_deps_times_sym_grad_u;
                      }
                    else
                      {
                        for (unsigned int d = 0; d < dim; ++d)
                          velocity_terms[d][d] -= 2.0 * one_third * trace(cell_data->strain_rate_table(cell,q)) * deta_deps_times_sym_grad_u;
                      }

                    if (cell_data->enable_prescribed_dilation)
                      {
                        pressure_terms += ( ( cell_data->dilation_derivative_wrt_strain_rate_table(cell,q)
                                              * sym_grad_u )
                                            +
                                            ( cell_data->dilation_derivative_wrt_pressure_table(cell,q)
                                              * cell_data->pressure_scaling * val_p )
                                          )
                                          * cell_data->pressure_scaling;
                      }
                  }
              }

            u_eval.submit_symmetric_gradient(velocity_terms, q);
            p_eval.submit_value(pressure_terms, q);
          }

        u_eval.integrate_scatter(EvaluationFlags::gradients, dst.block(0));
        p_eval.integrate_scatter(EvaluationFlags::values, dst.block(1));
      }
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::StokesOperator<dim, degree_v, number>
  ::local_apply_face(const dealii::MatrixFree<dim, number> &,
                     dealii::LinearAlgebra::distributed::BlockVector<number> &,
                     const dealii::LinearAlgebra::distributed::BlockVector<number> &,
                     const std::pair<unsigned int, unsigned int> &) const
  {
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::StokesOperator<dim, degree_v, number>
  ::local_apply_boundary_face(const dealii::MatrixFree<dim, number> &data,
                              dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                              const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
                              const std::pair<unsigned int, unsigned int> &face_range) const
  {
    // Assemble the fictive stabilization stress (phi_u[i].g)*(phi_u[j].n)
    // g=pressure_perturbation * g_hat is stored in free_surface_stabilization_term_table
    //  n is the normal vector
    FEFaceEvaluation<dim, degree_v, degree_v + 1, dim, number> velocity(data);
    const unsigned int n_faces_interior = data.n_inner_face_batches();

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        const auto boundary_id = data.get_boundary_id(face);
        if (cell_data->free_surface_boundary_indicators.find(boundary_id)
            == cell_data->free_surface_boundary_indicators.end())
          continue;

        velocity.reinit(face);
        velocity.gather_evaluate (src.block(0), EvaluationFlags::values);

        for (const unsigned int q : velocity.quadrature_point_indices())
          {
            const Tensor<1, dim, VectorizedArray<number>> phi_u_i = velocity.get_value(q);
#if DEAL_II_VERSION_GTE(9,7,0)
            const auto &normal_vector = velocity.normal_vector(q);
#else
            const auto &normal_vector = velocity.get_normal_vector(q);
#endif
            const auto stabilization_tensor = cell_data->free_surface_stabilization_term_table(face - n_faces_interior, q);
            const auto value_submit = -(stabilization_tensor * phi_u_i) * normal_vector;

            velocity.submit_value(value_submit, q);
          }
        velocity.integrate_scatter(EvaluationFlags::values, dst.block(0));
      }
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::StokesOperator<dim,degree_v,number>
  ::apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
               const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const
  {
    if (cell_data->apply_stabilization_free_surface_faces)
      MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number>>::
      data->loop(&StokesOperator::local_apply,
                 &StokesOperator::local_apply_face,
                 &StokesOperator::local_apply_boundary_face,
                 this,
                 dst,
                 src,
                 false, /*zero_dst_vector*/
                 MatrixFree<dim, number>::DataAccessOnFaces::values,
                 MatrixFree<dim, number>::DataAccessOnFaces::values);

    else
      MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number>>::
      data->cell_loop(&StokesOperator::local_apply, this, dst, src);
  }

  template <int dim, int degree_v, typename number>
  MatrixFreeStokesOperators::BTBlockOperator<dim,degree_v,number>::BTBlockOperator ()
    :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number>>()
  {}



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BTBlockOperator<dim,degree_v,number>::clear ()
  {
    this->cell_data = nullptr;
    MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::BlockVector<number>>::clear();
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BTBlockOperator<dim,degree_v,number>::
  set_cell_data (const OperatorCellData<dim,number> &data)
  {
    this->cell_data = &data;
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BTBlockOperator<dim,degree_v,number>
  ::compute_diagonal ()
  {
    // There is no need in the code for this diagonal.
    Assert(false, ExcNotImplemented());
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BTBlockOperator<dim,degree_v,number>
  ::local_apply (const dealii::MatrixFree<dim, number>                         &data,
                 dealii::LinearAlgebra::distributed::BlockVector<number>       &dst,
                 const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
                 const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim,degree_v,degree_v+1,dim,number> u_eval(data, 0);
    FEEvaluation<dim,degree_v-1,degree_v+1,1,number> p_eval(data, /*dofh*/1);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        u_eval.reinit(cell);

        p_eval.reinit(cell);
        p_eval.gather_evaluate(src.block(1), EvaluationFlags::values);

        for (const unsigned int q : u_eval.quadrature_point_indices())
          {
            const VectorizedArray<number> val_p = p_eval.get_value(q);


            SymmetricTensor<2,dim,VectorizedArray<number>>
            velocity_terms;

            for (unsigned int d=0; d<dim; ++d)
              velocity_terms[d][d] -= cell_data->pressure_scaling * val_p;
            u_eval.submit_symmetric_gradient(velocity_terms, q);
          }

        u_eval.integrate_scatter(EvaluationFlags::gradients, dst.block(0));
      }
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BTBlockOperator<dim, degree_v, number>
  ::local_apply_face(const dealii::MatrixFree<dim, number> &,
                     dealii::LinearAlgebra::distributed::BlockVector<number> &,
                     const dealii::LinearAlgebra::distributed::BlockVector<number> &,
                     const std::pair<unsigned int, unsigned int> &) const
  {
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BTBlockOperator<dim,degree_v,number>
  ::apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
               const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const
  {
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number>>::
    data->cell_loop(&BTBlockOperator::local_apply, this, dst, src);
  }



  /**
   *Operator for B block.
   */
  template <int dim, int degree_v, typename number>
  MatrixFreeStokesOperators::BBlockOperator<dim,degree_v,number>::BBlockOperator ():
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number>>()
  {}

  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BBlockOperator<dim,degree_v,number>::clear ()
  {
    this->cell_data = nullptr;
    MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::BlockVector<number>>::clear();
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BBlockOperator<dim,degree_v,number>::
  set_cell_data (const OperatorCellData<dim,number> &data)
  {
    this->cell_data = &data;
  }


  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BBlockOperator<dim,degree_v,number>
  ::compute_diagonal ()
  {
    // There is no need in the code for this diagonal.
    Assert(false, ExcNotImplemented());
  }



  template<int dim,int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BBlockOperator<dim,degree_v,number>
  ::local_apply(const dealii::MatrixFree<dim,number> &data,
                dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
                const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,degree_v,degree_v+1,dim,number> u_eval(data, 0);
    FEEvaluation<dim,degree_v-1,degree_v+1,1,number> p_eval(data, /*dofh*/1);



    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {

        u_eval.reinit(cell);
        u_eval.gather_evaluate(src.block(0), EvaluationFlags::gradients);

        p_eval.reinit(cell);


        SymmetricTensor<2,dim,VectorizedArray<number>> sym_grad_u;

        for (const unsigned int q : u_eval.quadrature_point_indices())
          {
            sym_grad_u = u_eval.get_symmetric_gradient(q);

            const VectorizedArray<number> div_u = trace(sym_grad_u);
            VectorizedArray<number> pressure_terms =
              -cell_data->pressure_scaling * div_u;
            p_eval.submit_value(pressure_terms, q);



          }


        p_eval.integrate_scatter(EvaluationFlags::values, dst.block(1));
      }
  }

  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::BBlockOperator<dim,degree_v,number>
  ::apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
               const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const
  {
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number>>::
    data->cell_loop(&BBlockOperator::local_apply, this, dst, src);
  }

  /**
   * Mass matrix operator on pressure
   */
  template <int dim, int degree_p, typename number>
  MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>::MassMatrixOperator ()
    :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>()
  {}



  template <int dim, int degree_p, typename number>
  void
  MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>::clear ()
  {
    this->cell_data = nullptr;
    MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number>>::clear();
  }



  template <int dim, int degree_p, typename number>
  void
  MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>::reinit(const Mapping<dim>              &mapping,
                                                                             const DoFHandler<dim>           &dof_handler_v,
                                                                             const DoFHandler<dim>           &dof_handler_p,
                                                                             const AffineConstraints<number> &constraints_v,
                                                                             const AffineConstraints<number> &constraints_p,
                                                                             std::shared_ptr<MatrixFree<dim,double>> mf_storage,
                                                                             const unsigned int level)
  {
    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags =
      update_quadrature_points /*| update_gradients*/ | update_values;
    data.mg_level = level;

    data.tasks_parallel_scheme =
      MatrixFree<dim,double>::AdditionalData::none;
    AffineConstraints<number> dummy;

    mf_storage->reinit(mapping,
    std::vector< const DoFHandler< dim > *> {&dof_handler_v, &dof_handler_p},
    std::vector< const AffineConstraints< number > *> {&constraints_v, &constraints_p} ,
    QGauss<1>(degree_p+2), data);

    this->initialize(mf_storage, std::vector< unsigned int > {1}, std::vector< unsigned int > {1});
  }



  template <int dim, int degree_p, typename number>
  void
  MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>::
  set_cell_data (const OperatorCellData<dim,number> &data)
  {
    this->cell_data = &data;
  }



  template <int dim, int degree_p, typename number>
  void
  MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>
  ::local_apply (const dealii::MatrixFree<dim, number>                 &data,
                 dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                 const dealii::LinearAlgebra::distributed::Vector<number> &src,
                 const std::pair<unsigned int, unsigned int>           &cell_range) const
  {
    FEEvaluation<dim,degree_p,degree_p+2,1,number> pressure (data, /*dofh*/1);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        pressure.reinit (cell);
        pressure.gather_evaluate (src, EvaluationFlags::values);
        this->inner_cell_operation(pressure);
        pressure.integrate_scatter (EvaluationFlags::values, dst);
      }

  }


  template <int dim, int degree_p, typename number>
  void
  MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>
  ::inner_cell_operation(FEEvaluation<dim,
                         degree_p,
                         degree_p+2,
                         1,
                         number> &pressure) const
  {
    const bool use_viscosity_at_quadrature_points
      = (cell_data->viscosity.size(1) == pressure.n_q_points);

    const unsigned int cell = pressure.get_current_cell_index();
    const unsigned int n_components_filled = this->get_matrix_free()->n_active_entries_per_cell_batch(cell);

    VectorizedArray<number> prefactor;

    // The /= operator for VectorizedArray results in a floating point operation
    // (divide by 0) since the (*viscosity)(cell) array is not completely filled.
    // Therefore, we need to divide each entry manually.
    if (!use_viscosity_at_quadrature_points)
      {
        for (unsigned int c=0; c<n_components_filled; ++c)
          prefactor[c] = cell_data->pressure_scaling*cell_data->pressure_scaling / cell_data->viscosity(cell, 0)[c];
      }

    for (const unsigned int q : pressure.quadrature_point_indices())
      {
        // Only update the viscosity if a Q1 projection is used.
        if (use_viscosity_at_quadrature_points)
          {
            for (unsigned int c=0; c<n_components_filled; ++c)
              prefactor[c] = cell_data->pressure_scaling*cell_data->pressure_scaling / cell_data->viscosity(cell, q)[c];
          }

        pressure.submit_value(prefactor*pressure.get_value(q), q);
      }
  }



  template <int dim, int degree_p, typename number>
  void
  MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>
  ::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
               const dealii::LinearAlgebra::distributed::Vector<number> &src) const
  {
    MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number>>::
    data->cell_loop(&MassMatrixOperator::local_apply, this, dst, src);
  }



  template <int dim, int degree_p, typename number>
  void
  MatrixFreeStokesOperators::MassMatrixOperator<dim,degree_p,number>
  ::compute_diagonal ()
  {
    this->inverse_diagonal_entries =
      std::make_shared<DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>>();
    this->diagonal_entries =
      std::make_shared<DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>>();

    dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    dealii::LinearAlgebra::distributed::Vector<number> &diagonal =
      this->diagonal_entries->get_vector();

    this->data->initialize_dof_vector(inverse_diagonal, /*dofh*/1);
    this->data->initialize_dof_vector(diagonal, /*dofh*/1);

    MatrixFreeTools::compute_diagonal<dim,degree_p,degree_p+2,1,number,VectorizedArray<number>,dealii::LinearAlgebra::distributed::Vector<number>>(
      *(this->get_matrix_free()),
      diagonal,
      [&](FEEvaluation<dim,
          degree_p,
          degree_p+2,
          1,
          number> &pressure)
    {
      pressure.evaluate(EvaluationFlags::values);
      this->inner_cell_operation(pressure);
      pressure.integrate(EvaluationFlags::values);
    },
    1 /* dofhandler */);


    this->set_constrained_entries_to_one(diagonal);
    inverse_diagonal = diagonal;

    // Finally loop over all of the computed diagonal elements and invert them.
    // The following loop relies on the fact that inverse_diagonal.begin()/end()
    // iterates only over the *locally owned* elements of the vector in which
    // we store inverse_diagonal.
    for (auto &local_element : inverse_diagonal)
      {
        Assert(local_element > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero or negative."));
        local_element = 1./local_element;
      }
  }

  template<class BOperatorType, class BTOperatorType>
  void MatrixFreeStokesOperators::BC_invBT_Operator<BOperatorType, BTOperatorType>::vmult(dealii::LinearAlgebra::distributed::Vector<double> &dst,
      const dealii::LinearAlgebra::distributed::Vector<double> &src) const
  {
    dealii::LinearAlgebra::distributed::BlockVector<double> block_src;
    dealii::LinearAlgebra::distributed::BlockVector<double> block_dst;

    const auto &B_matrix_free=*B_operator.get_matrix_free();
    block_src.reinit(2);
    block_dst.reinit(2);

    B_matrix_free.initialize_dof_vector(block_src.block(0),0);
    B_matrix_free.initialize_dof_vector(block_src.block(1),1);

    B_matrix_free.initialize_dof_vector(block_dst.block(0),0);
    B_matrix_free.initialize_dof_vector(block_dst.block(1),1);

    block_src.block(1)=src;
    block_src.block(0)=0;
    block_dst=0;
    BT_operator.vmult(block_dst,block_src);

    block_dst.block(0).scale(diag_A_inv);

    block_src.block(0)=block_dst.block(0);
    block_src.block(1)=0;
    block_dst=0;
    B_operator.vmult(block_dst,block_src);
    dst=block_dst.block(1);
  }


  template <int dim, int degree_p, typename number, typename BOperatorType, typename BTOperatorType>
  MatrixFreeStokesOperators::PressureLaplaceOperator<dim, degree_p, number, BOperatorType, BTOperatorType>::PressureLaplaceOperator ()
    :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>(),
    cell_data(nullptr)
  {}



  template <int dim, int degree_p, typename number, typename BOperatorType, typename BTOperatorType>
  void
  MatrixFreeStokesOperators::PressureLaplaceOperator<dim, degree_p, number, BOperatorType, BTOperatorType>::clear ()
  {
    this->cell_data = nullptr;
    this->BCinvBTOperator.reset();
    MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number>>::clear();
  }



  template <int dim, int degree_p, typename number, typename BOperatorType, typename BTOperatorType>
  void
  MatrixFreeStokesOperators::PressureLaplaceOperator<dim, degree_p, number, BOperatorType, BTOperatorType>::reinit(const Mapping<dim>              &mapping,
      const DoFHandler<dim>           &dof_handler_v,
      const DoFHandler<dim>           &dof_handler_p,
      const AffineConstraints<number> &constraints_v,
      const AffineConstraints<number> &constraints_p,
      std::shared_ptr<MatrixFree<dim,double>> mf_storage,
      const unsigned int level)
  {
    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags =
      update_quadrature_points /*| update_gradients*/ | update_values;
    data.mg_level = level;

    data.tasks_parallel_scheme =
      MatrixFree<dim,double>::AdditionalData::none;
    AffineConstraints<number> dummy;

    mf_storage->reinit(mapping,
    std::vector< const DoFHandler< dim > *> {&dof_handler_v, &dof_handler_p},
    std::vector< const AffineConstraints< number > *> {&constraints_v, &constraints_p} ,
    QGauss<1>(degree_p+2), data);

    this->initialize(mf_storage, std::vector< unsigned int > {1}, std::vector< unsigned int > {1});
  }



  template <int dim, int degree_p, typename number, typename BOperatorType, typename BTOperatorType>
  void
  MatrixFreeStokesOperators::PressureLaplaceOperator<dim, degree_p, number, BOperatorType, BTOperatorType>::
  set_cell_data (const OperatorCellData<dim,number> &data)
  {
    this->cell_data = &data;
  }



  template <int dim, int degree_p, typename number, typename BOperatorType, typename BTOperatorType>
  void
  MatrixFreeStokesOperators::PressureLaplaceOperator<dim, degree_p, number, BOperatorType, BTOperatorType>
  ::local_apply (const dealii::MatrixFree<dim, number>                 &data,
                 dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                 const dealii::LinearAlgebra::distributed::Vector<number> &src,
                 const std::pair<unsigned int, unsigned int>           &cell_range) const
  {
    FEEvaluation<dim,degree_p,degree_p+2,1,number> pressure (data, /*dofh*/1);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        pressure.reinit (cell);
        pressure.gather_evaluate (src, EvaluationFlags::gradients);
        this->inner_cell_operation(pressure);
        pressure.integrate_scatter (EvaluationFlags::gradients, dst);
      }

  }


  template <int dim, int degree_p, typename number, typename BOperatorType, typename BTOperatorType>
  void
  MatrixFreeStokesOperators::PressureLaplaceOperator<dim, degree_p, number, BOperatorType, BTOperatorType>
  ::inner_cell_operation(FEEvaluation<dim,
                         degree_p,
                         degree_p+2,
                         1,
                         number> &pressure) const
  {
    const bool use_viscosity_at_quadrature_points
      = (cell_data->viscosity.size(1) == pressure.n_q_points);

    const unsigned int cell = pressure.get_current_cell_index();
    const unsigned int n_components_filled = this->get_matrix_free()->n_active_entries_per_cell_batch(cell);

    VectorizedArray<number> prefactor;

    // The /= operator for VectorizedArray results in a floating point operation
    // (divide by 0) since the (*viscosity)(cell) array is not completely filled.
    // Therefore, we need to divide each entry manually.
    if (!use_viscosity_at_quadrature_points)
      {
        for (unsigned int c=0; c<n_components_filled; ++c)
          prefactor[c] = cell_data->pressure_scaling*cell_data->pressure_scaling / cell_data->viscosity(cell, 0)[c];
      }

    for (const unsigned int q : pressure.quadrature_point_indices())
      {
        // Only update the viscosity if a Q1 projection is used.
        if (use_viscosity_at_quadrature_points)
          {
            for (unsigned int c=0; c<n_components_filled; ++c)
              prefactor[c] = cell_data->pressure_scaling*cell_data->pressure_scaling / cell_data->viscosity(cell, q)[c];
          }

        pressure.submit_gradient(prefactor*pressure.get_gradient(q), q);
      }
  }



  template <int dim, int degree_p, typename number, typename BOperatorType, typename BTOperatorType>
  void
  MatrixFreeStokesOperators::PressureLaplaceOperator<dim, degree_p, number, BOperatorType, BTOperatorType>
  ::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
               const dealii::LinearAlgebra::distributed::Vector<number> &src) const
  {
    if (BCinvBTOperator)
      {
        dealii::LinearAlgebra::distributed::Vector<number> tmp;
        this->data->initialize_dof_vector(tmp,1);
        BCinvBTOperator->vmult(tmp, src);
        dst+=tmp;
      }
    else
      MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number>>::
      data->cell_loop(&PressureLaplaceOperator::local_apply, this, dst, src);
  }



  template <int dim, int degree_p, typename number, typename BOperatorType, typename BTOperatorType>
  void
  MatrixFreeStokesOperators::PressureLaplaceOperator<dim, degree_p, number, BOperatorType, BTOperatorType>
  ::compute_diagonal ()
  {
    this->inverse_diagonal_entries =
      std::make_shared<DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>>();
    this->diagonal_entries =
      std::make_shared<DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>>();

    dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    dealii::LinearAlgebra::distributed::Vector<number> &diagonal =
      this->diagonal_entries->get_vector();

    this->data->initialize_dof_vector(inverse_diagonal, /*dofh*/1);
    this->data->initialize_dof_vector(diagonal, /*dofh*/1);

    MatrixFreeTools::compute_diagonal<dim,degree_p,degree_p+2,1,number,VectorizedArray<number>,dealii::LinearAlgebra::distributed::Vector<number>>(
      *(this->get_matrix_free()),
      diagonal,
      [&](FEEvaluation<dim,
          degree_p,
          degree_p+2,
          1,
          number> &pressure)
    {
      pressure.evaluate(EvaluationFlags::gradients);
      this->inner_cell_operation(pressure);
      pressure.integrate(EvaluationFlags::gradients);
    },
    1 /* dofhandler */);
    this->set_constrained_entries_to_one(diagonal);
    inverse_diagonal = diagonal;

    // Finally loop over all of the computed diagonal elements and invert them.
    // The following loop relies on the fact that inverse_diagonal.begin()/end()
    // iterates only over the *locally owned* elements of the vector in which
    // we store inverse_diagonal.
    for (auto &local_element : inverse_diagonal)
      {
        Assert(local_element > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero or negative."));
        local_element = 1./local_element;
      }
  }



  template<int dim, int degree_p, typename number, typename BOperatorType,
           typename BTOperatorType>
  void MatrixFreeStokesOperators::PressureLaplaceOperator<dim, degree_p, number, BOperatorType, BTOperatorType>::set_BCinvBT(const BOperatorType &B, const BTOperatorType &BT, const dealii::LinearAlgebra::distributed::Vector<double> &inverse_lumped_velocity_mass_matrix)
  {
    this->BCinvBTOperator=std::make_unique<BC_invBT_Operator<BOperatorType,
          BTOperatorType>>(B,BT,inverse_lumped_velocity_mass_matrix);
  }




  /**
   * Velocity block operator
   */
  template <int dim, int degree_v, typename number>
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>::ABlockOperator ()
    :
    MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>()
  {}



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>::clear ()
  {
    this->cell_data = nullptr;
    MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number>>::clear();
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>::reinit(const Mapping<dim>              &mapping,
                                                                         const DoFHandler<dim>           &dof_handler_v,
                                                                         const DoFHandler<dim>           &dof_handler_p,
                                                                         const AffineConstraints<number> &constraints_v,
                                                                         const AffineConstraints<number> &constraints_p,
                                                                         std::shared_ptr<MatrixFree<dim,double>> mf_storage,
                                                                         const unsigned int level)
  {
    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_quadrature_points | update_gradients | update_values;
    data.mg_level = level;

    typename MatrixFree<dim,double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim,double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_values | update_gradients |
                                            update_JxW_values | update_quadrature_points);

    AffineConstraints<number> dummy;
    mf_storage->reinit(mapping,
    std::vector< const DoFHandler< dim > *> {&dof_handler_v, &dof_handler_p},
    std::vector< const AffineConstraints< number > *> {&constraints_v, &constraints_p} ,
    QGauss<1>(degree_v+1), additional_data);

    this->initialize(mf_storage, std::vector< unsigned int > {0}, std::vector< unsigned int > {0});
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>::
  set_cell_data (const OperatorCellData<dim,number> &data)
  {
    this->cell_data = &data;
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
  ::inner_cell_operation(FEEvaluation<dim,
                         degree_v,
                         degree_v+1,
                         dim,
                         number> &velocity) const
  {
    const bool use_viscosity_at_quadrature_points
      = (cell_data->viscosity.size(1) == velocity.n_q_points);

    const unsigned int cell = velocity.get_current_cell_index();
    VectorizedArray<number> viscosity_x_2 = 2.0*cell_data->viscosity(cell, 0);

    for (const unsigned int q : velocity.quadrature_point_indices())
      {
        // Only update the viscosity if a Q1 projection is used.
        if (use_viscosity_at_quadrature_points)
          viscosity_x_2 = 2.0*cell_data->viscosity(cell, q);

        SymmetricTensor<2,dim,VectorizedArray<number>> sym_grad_u =
          velocity.get_symmetric_gradient (q);
        sym_grad_u *= viscosity_x_2;

        if (cell_data->is_compressible ||
            cell_data->enable_prescribed_dilation)
          {
            const VectorizedArray<number> div = trace(sym_grad_u);
            for (unsigned int d=0; d<dim; ++d)
              sym_grad_u[d][d] -= 1.0/3.0*div;
          }

        velocity.submit_symmetric_gradient(sym_grad_u, q);
      }
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
  ::cell_operation(FEEvaluation<dim,
                   degree_v,
                   degree_v+1,
                   dim,
                   number> &velocity) const
  {
    velocity.evaluate (EvaluationFlags::gradients);
    this->inner_cell_operation(velocity);
    velocity.integrate(EvaluationFlags::gradients);
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
  ::local_apply (const dealii::MatrixFree<dim, number>                 &data,
                 dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                 const dealii::LinearAlgebra::distributed::Vector<number> &src,
                 const std::pair<unsigned int, unsigned int>           &cell_range) const
  {
    FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (data,0);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        velocity.reinit (cell);

        // Instead of calling
        //   velocity.read_dof_values (src);
        //   velocity.evaluate (EvaluationFlags::gradients);
        // (the latter by calling cell_operation()), we use the more efficient
        // combined gather_evaluate() and use inner_cell_operation().
        velocity.gather_evaluate (src, EvaluationFlags::gradients);
        this->inner_cell_operation(velocity);
        velocity.integrate_scatter (EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
  ::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
               const dealii::LinearAlgebra::distributed::Vector<number> &src) const
  {
    MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number>>::
    data->cell_loop(&ABlockOperator::local_apply, this, dst, src);
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
  ::compute_diagonal ()
  {
    this->inverse_diagonal_entries =
      std::make_shared<DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>>();
    dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);

    MatrixFreeTools::compute_diagonal(
      *(this->get_matrix_free()),
      inverse_diagonal,
      &MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>::cell_operation,
      this);

    this->set_constrained_entries_to_one(inverse_diagonal);

    // Finally loop over all of the computed diagonal elements and invert them.
    // The following loop relies on the fact that inverse_diagonal.begin()/end()
    // iterates only over the *locally owned* elements of the vector in which
    // we store inverse_diagonal.
    for (auto &local_element : inverse_diagonal)
      {
        Assert(local_element > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero or negative."));
        local_element = 1./local_element;
      }
  }



  template <int dim, int degree_v, typename number>
  void
  MatrixFreeStokesOperators::ABlockOperator<dim,degree_v,number>
  ::set_diagonal (const dealii::LinearAlgebra::distributed::Vector<number> &diag)
  {
    this->inverse_diagonal_entries =
      std::make_shared<DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>>();
    dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);

    inverse_diagonal = diag;

    this->set_constrained_entries_to_one(inverse_diagonal);

    // Finally loop over all of the computed diagonal elements and invert them.
    // The following loop relies on the fact that inverse_diagonal.begin()/end()
    // iterates only over the *locally owned* elements of the vector in which
    // we store inverse_diagonal.
    for (auto &local_element : inverse_diagonal)
      {
        Assert(local_element > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero or negative."));
        local_element = 1./local_element;
      }
  }

  template<typename Range,
           typename Domain,
           typename Payload>
  LinearOperator<Range, Domain, Payload> remove_mean_value(LinearOperator<Range,Domain,Payload> &exemplar)
  {
    LinearOperator<Range, Domain, Payload> return_op;

    return_op.reinit_range_vector  = exemplar.reinit_range_vector;
    return_op.reinit_domain_vector = exemplar.reinit_domain_vector;

    return_op.vmult = [&](Range &dest, const Domain &src)
    {

      dest = src;
      dest.add(-dest.mean_value());
    };
    return return_op;
  }

// Do not include entries associated with constrained dofs in average
// for mean value removal.
  template <typename VectorType>
  void remove_mean_value_free_dofs(VectorType &v,
                                   const dealii::AffineConstraints<double> &constraints)
  {
    const auto &part = *v.get_partitioner();
    double sum = 0.;
    dealii::types::global_dof_index n = 0;
    for (unsigned int i = 0; i < v.locally_owned_size(); ++i)
      if (!constraints.is_constrained(part.local_to_global(i)))
        {
          sum += v.local_element(i);
          ++n;
        }
    sum = dealii::Utilities::MPI::sum(sum, v.get_mpi_communicator());
    n   = dealii::Utilities::MPI::sum(n,   v.get_mpi_communicator());
    const double mean = sum / n;
    for (unsigned int i = 0; i < v.locally_owned_size(); ++i)
      if (!constraints.is_constrained(part.local_to_global(i)))
        v.local_element(i) -= mean;
  }
  template <class StokesMatrixType, class AOperatorType, class BOperatorType, class BTOperatorType,class SchurComplementMatrixType, class VectorType, class PreconditionerMp>
  MatrixFreeStokesOperators::DiagBFBT<StokesMatrixType, AOperatorType, BOperatorType, BTOperatorType, SchurComplementMatrixType, VectorType, PreconditionerMp>::DiagBFBT(
    const PreconditionerMp &mp_preconditioner,
    const PreconditionerMp &mp_preconditioner_damped,
    const bool do_solve_schur_complement,
    const double solver_tolerance,
    const dealii::LinearAlgebra::distributed::Vector<double> &lumped_velocity_mass_matrix,
    const dealii::LinearAlgebra::distributed::Vector<double> &lumped_velocity_mass_matrix_damped,
    const StokesMatrixType &system_matrix,
    const AOperatorType &A_operator,
    const BOperatorType &B_operator,
    const BTOperatorType &BT_operator,
    const SchurComplementMatrixType &mp_matrix,
    const dealii::AffineConstraints<double> &constraints,
    const bool pressure_laplace_has_nullspace)
    : n_iterations_(0),
      mp_preconditioner(mp_preconditioner),
      mp_preconditioner_damped(mp_preconditioner_damped),
      do_solve_schur_complement(do_solve_schur_complement),
      solver_tolerance(solver_tolerance),
      lumped_velocity_mass_matrix(lumped_velocity_mass_matrix),
      lumped_velocity_mass_matrix_damped(lumped_velocity_mass_matrix_damped),
      system_matrix(system_matrix),
      A_operator(A_operator),
      B_operator(B_operator),
      BT_operator(BT_operator),
      mp_matrix(mp_matrix),
      constraints(constraints),
      pressure_laplace_has_nullspace(pressure_laplace_has_nullspace)
  {
  }

  template <class StokesMatrixType, class AOperatorType, class BOperatorType, class BTOperatorType, class SchurComplementMatrixType, class VectorType, class PreconditionerMp>
  void MatrixFreeStokesOperators::DiagBFBT<StokesMatrixType, AOperatorType, BOperatorType, BTOperatorType, SchurComplementMatrixType, VectorType, PreconditionerMp>::vmult(
    VectorType &dst, const VectorType &src) const
  {


    try
      {


        MatrixFreeStokesOperators::BC_invBT_Operator<BOperatorType, BTOperatorType>
        Op_BC_invBT(B_operator, BT_operator, lumped_velocity_mass_matrix);
        dealii::LinearOperator<VectorType> op_BC_invBT;


        op_BC_invBT.reinit_range_vector=[&](VectorType &v, bool)
        {
          v.reinit(src);
        };
        op_BC_invBT.reinit_domain_vector=[&](VectorType &v, bool)
        {
          v.reinit(src);
        };
        op_BC_invBT.vmult=[&](VectorType &dst, const VectorType &src)
        {
          Op_BC_invBT.vmult(dst,src);
        };


        dealii::LinearOperator<VectorType> op_mp_preconditioner;
        op_mp_preconditioner.reinit_range_vector=[&](VectorType &v, bool)
        {
          v.reinit(src);
        };
        op_mp_preconditioner.reinit_domain_vector=[&](VectorType &v, bool)
        {
          v.reinit(src);
        };

        //precondition with solve
        op_mp_preconditioner.vmult=[&](VectorType &dst, const VectorType &src)
        {
          dst=0.0;
          for (unsigned int i = 0; i<2; ++i)
            {

              VectorType tmp;
              VectorType residual=src;
              tmp.reinit(src);
              tmp=0.0;
              Op_BC_invBT.vmult(tmp,dst);
              residual-=tmp;
              VectorType corrected;
              corrected.reinit(dst);
              mp_preconditioner.vmult(corrected,residual);
              dst+=corrected;
              dst.add(-dst.mean_value());
            }


        };
        auto remove_pressure_laplace_nullspace=[&](VectorType &v)
        {
          if (pressure_laplace_has_nullspace)
            remove_mean_value_free_dofs(v,constraints);
        };
        auto rmv=remove_mean_value<>(op_BC_invBT);
        VectorType ptmp;
        VectorType ptmp2;
        ptmp.reinit(src);
        ptmp2.reinit(src);
        PrimitiveVectorMemory<VectorType> mem;



        VectorType rhs1=src; //nullspace removal
        //       rhs1.add(-rhs1.mean_value());
        remove_pressure_laplace_nullspace(rhs1);



//          SolverControl solver_control(5000, rhs1.l2_norm() * solver_tolerance, false, true);
//         SolverCG<VectorType> solver(solver_control,mem);

        ptmp = 0;

        mp_preconditioner_damped.vmult(ptmp,rhs1);
        //        ptmp.add(-ptmp.mean_value());
        remove_pressure_laplace_nullspace(ptmp);


//           solver.solve(rmv*op_BC_invBT_damped, ptmp, rhs1, mp_preconditioner);
//          n_iterations_ += solver_control.last_step();

        {
          dealii::LinearAlgebra::distributed::BlockVector<double> block_src;
          dealii::LinearAlgebra::distributed::BlockVector<double> block_dst;
          block_src.reinit(2);
          block_dst.reinit(2);
          system_matrix.initialize_dof_vector(block_src);
          system_matrix.initialize_dof_vector(block_dst);

          block_src.block(1) = ptmp;
          block_src.block(0) = 0;
          block_dst = 0;
          BT_operator.vmult(block_dst, block_src);

          block_dst.block(0).scale(lumped_velocity_mass_matrix_damped);

          A_operator.vmult(block_src.block(0), block_dst.block(0));

          block_src.block(0).scale(lumped_velocity_mass_matrix);

          block_src.block(1) = 0;
          block_dst = 0;
          B_operator.vmult(block_dst, block_src);
          ptmp2 = block_dst.block(1);
        }

        VectorType rhs2=ptmp2;
//         rhs2.add(-rhs2.mean_value());

        remove_pressure_laplace_nullspace(rhs2);


        dst = 0;
        mp_preconditioner.vmult(dst,rhs2);
//          if (std::abs(dst.mean_value())>(1e-6*rhs2.l2_norm()))
//            {
//              std::cout<<"dst mean value is "<<dst.mean_value();
//            }
//          solver_control.set_tolerance(solver_tolerance*rhs2.l2_norm());
//          solver.solve(rmv*op_BC_invBT, dst, rhs2, mp_preconditioner);

//          dst.add(-dst.mean_value());

        remove_pressure_laplace_nullspace(dst);
//         n_iterations_ += solver_control.last_step();


      }

    catch (const std::exception &exc)
      {
        Utilities::throw_linear_solver_failure_exception("iterative (DiagBFBT) solver",
                                                         "DiagBFBT::vmult",
                                                         std::vector<SolverControl> {},
                                                         exc,
                                                         src.get_mpi_communicator());
      }
  }

  template <class StokesMatrixType, class AOperatorType, class BOperatorType, class BTOperatorType, class SchurComplementMatrixType, class VectorType, class PreconditionerMp>
  unsigned int MatrixFreeStokesOperators::DiagBFBT<StokesMatrixType, AOperatorType, BOperatorType, BTOperatorType, SchurComplementMatrixType, VectorType, PreconditionerMp>::n_iterations() const
  {
    return n_iterations_;
  }

}









// explicit instantiationsdealii.mak
namespace aspect
{
#define INSTANTIATE(dim) \
  template class MatrixFreeStokesOperators::ABlockOperator<dim,2,GMGNumberType>; \
  template class MatrixFreeStokesOperators::ABlockOperator<dim,3,GMGNumberType>; \
  template class MatrixFreeStokesOperators::StokesOperator<dim,2,GMGNumberType>; \
  template class MatrixFreeStokesOperators::StokesOperator<dim,3,GMGNumberType>; \
  template class MatrixFreeStokesOperators::BTBlockOperator<dim,2,GMGNumberType>; \
  template class MatrixFreeStokesOperators::BTBlockOperator<dim,3,GMGNumberType>; \
  template class MatrixFreeStokesOperators::BBlockOperator<dim,2,GMGNumberType>;\
  template class MatrixFreeStokesOperators::BBlockOperator<dim,3,GMGNumberType>;\
  template class MatrixFreeStokesOperators::MassMatrixOperator<dim,1,GMGNumberType>; \
  template class MatrixFreeStokesOperators::MassMatrixOperator<dim,2,GMGNumberType>; \
  template struct MatrixFreeStokesOperators::OperatorCellData<dim, GMGNumberType>; \
  template void MatrixFreeStokesOperators::fill_active_cell_data<dim, GMGNumberType>( \
      const DoFHandler<dim> &, \
      const DoFHandler<dim> &, \
      const Introspection<dim> &, \
      const Quadrature<dim> &, \
      const MaterialModel::Interface<dim> &, \
      const MaterialModel::MaterialAveraging::AveragingOperation &, \
      const Mapping<dim> &, \
      const MatrixFree<dim, double> &, \
      const MatrixFree<dim, double> &, \
      const MatrixFree<dim, double> &, \
      const LinearAlgebra::BlockVector &, \
      const MPI_Comm &, \
      dealii::LinearAlgebra::distributed::Vector<double> &, \
      MatrixFreeStokesOperators::OperatorCellData<dim, GMGNumberType> &, \
      double &, \
      double &); \
  template class MatrixFreeStokesOperators::PressureLaplaceOperator<dim,1,GMGNumberType, \
                                                                    MatrixFreeStokesOperators::BBlockOperator<dim,2,GMGNumberType>, \
                                                                    MatrixFreeStokesOperators::BTBlockOperator<dim,2,GMGNumberType>>; \
  template class MatrixFreeStokesOperators::PressureLaplaceOperator<dim,2,GMGNumberType, \
                                                                    MatrixFreeStokesOperators::BBlockOperator<dim,3,GMGNumberType>, \
                                                                    MatrixFreeStokesOperators::BTBlockOperator<dim,3,GMGNumberType>>; \
  template class MatrixFreeStokesOperators::LumpedVelocityMassOperator<dim,2,GMGNumberType>; \
  template class MatrixFreeStokesOperators::LumpedVelocityMassOperator<dim,3,GMGNumberType>; \
  template class MatrixFreeStokesOperators::BC_invBT_Operator<MatrixFreeStokesOperators::BBlockOperator<dim,2,GMGNumberType>, \
                                                              MatrixFreeStokesOperators::BTBlockOperator<dim,2,GMGNumberType>>; \
  template class MatrixFreeStokesOperators::BC_invBT_Operator<MatrixFreeStokesOperators::BBlockOperator<dim,3,GMGNumberType>, \
                                                              MatrixFreeStokesOperators::BTBlockOperator<dim,3,GMGNumberType>>; \
  template class MatrixFreeStokesOperators::DiagBFBT<MatrixFreeStokesOperators::StokesOperator<dim,2,double>, \
                                                     MatrixFreeStokesOperators::ABlockOperator<dim,2,double>, \
                                                     MatrixFreeStokesOperators::BBlockOperator<dim,2,double>, \
                                                     MatrixFreeStokesOperators::BTBlockOperator<dim,2,double>, \
                                                     MatrixFreeStokesOperators::MassMatrixOperator<dim,1,double>, \
                                                     dealii::LinearAlgebra::distributed::Vector<GMGNumberType>, \
                                                     PreconditionMG<dim, dealii::LinearAlgebra::distributed::Vector<GMGNumberType>, MGTransferType<dim,GMGNumberType>>>; \
  template class MatrixFreeStokesOperators::DiagBFBT<MatrixFreeStokesOperators::StokesOperator<dim,3,double>, \
                                                     MatrixFreeStokesOperators::ABlockOperator<dim,3,double>, \
                                                     MatrixFreeStokesOperators::BBlockOperator<dim,3,double>, \
                                                     MatrixFreeStokesOperators::BTBlockOperator<dim,3,double>, \
                                                     MatrixFreeStokesOperators::MassMatrixOperator<dim,2,double>, \
                                                     dealii::LinearAlgebra::distributed::Vector<GMGNumberType>, \
                                                     PreconditionMG<dim, dealii::LinearAlgebra::distributed::Vector<GMGNumberType>, MGTransferType<dim,GMGNumberType>>>; \
  template class internal::MGSmootherRemoveNullspace< \
                                                      dealii::LinearAlgebra::distributed::Vector<GMGNumberType>, \
                                                      dealii::mg::SmootherRelaxation< \
                                                          dealii::PreconditionChebyshev< \
                                                              MatrixFreeStokesOperators::PressureLaplaceOperator<dim,1,GMGNumberType, \
                                                                  MatrixFreeStokesOperators::BBlockOperator<dim,2,GMGNumberType>, \
                                                                  MatrixFreeStokesOperators::BTBlockOperator<dim,2,GMGNumberType>>, \
                                                              dealii::LinearAlgebra::distributed::Vector<GMGNumberType>>, \
                                                          dealii::LinearAlgebra::distributed::Vector<GMGNumberType>>>; \
  template class internal::MGSmootherRemoveNullspace< \
                                                      dealii::LinearAlgebra::distributed::Vector<GMGNumberType>, \
                                                      dealii::mg::SmootherRelaxation< \
                                                          dealii::PreconditionChebyshev< \
                                                              MatrixFreeStokesOperators::PressureLaplaceOperator<dim,2,GMGNumberType, \
                                                                  MatrixFreeStokesOperators::BBlockOperator<dim,3,GMGNumberType>, \
                                                                  MatrixFreeStokesOperators::BTBlockOperator<dim,3,GMGNumberType>>, \
                                                              dealii::LinearAlgebra::distributed::Vector<GMGNumberType>>, \
                                                          dealii::LinearAlgebra::distributed::Vector<GMGNumberType>>>; \

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE

}
