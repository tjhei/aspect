/*
  Copyright (C) 2011 - 2018 by the authors of the ASPECT code.

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


#include <aspect/stokes_matrix_free.h>
#include <aspect/global.h>
#include <aspect/citation_info.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/signaling_nan.h>
#include <deal.II/lac/solver_gmres.h>

namespace aspect
{
  namespace internal
  {
    namespace ChangeVectorTypes
    {
      void import(TrilinosWrappers::MPI::Vector &out,
                  const dealii::LinearAlgebra::ReadWriteVector<double> &rwv,
                  const VectorOperation::values                 operation)
      {
        Assert(out.size() == rwv.size(),
               ExcMessage("Both vectors need to have the same size for import() to work!"));

        Assert(out.locally_owned_elements() == rwv.get_stored_elements(),
               ExcNotImplemented());

        if (operation == VectorOperation::insert)
          {
            for (const auto idx : out.locally_owned_elements())
              out[idx] = rwv[idx];
          }
        else if (operation == VectorOperation::add)
          {
            for (const auto idx : out.locally_owned_elements())
              out[idx] += rwv[idx];
          }
        else
          AssertThrow(false, ExcNotImplemented());

        out.compress(operation);
      }


      void copy(TrilinosWrappers::MPI::Vector &out,
                const dealii::LinearAlgebra::distributed::Vector<double> &in)
      {
        dealii::LinearAlgebra::ReadWriteVector<double> rwv(out.locally_owned_elements());
        rwv.import(in, VectorOperation::insert);
        //This import function doesn't exist until after 9.0
        //Implemented above
        import(out, rwv,VectorOperation::insert);
      }

      void copy(dealii::LinearAlgebra::distributed::Vector<double> &out,
                const TrilinosWrappers::MPI::Vector &in)
      {
        dealii::LinearAlgebra::ReadWriteVector<double> rwv;
        rwv.reinit(in);
        out.import(rwv, VectorOperation::insert);
      }

      void copy(TrilinosWrappers::MPI::BlockVector &out,
                const dealii::LinearAlgebra::distributed::BlockVector<double> &in)
      {
        const unsigned int n_blocks = in.n_blocks();
        for (unsigned int b=0; b<n_blocks; ++b)
          copy(out.block(b),in.block(b));
      }

      void copy(dealii::LinearAlgebra::distributed::BlockVector<double> &out,
                const TrilinosWrappers::MPI::BlockVector &in)
      {
        const unsigned int n_blocks = in.n_blocks();
        for (unsigned int b=0; b<n_blocks; ++b)
          copy(out.block(b),in.block(b));
      }
    }

    /**
             * Implement the block Schur preconditioner for the Stokes system.
             */
    template <class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
    class BlockSchurPreconditioner : public Subscriptor
    {
      public:
        /**
                   * @brief Constructor
                   *
                   * @param S The entire Stokes matrix
                   * @param Spre The matrix whose blocks are used in the definition of
                   *     the preconditioning of the Stokes matrix, i.e. containing approximations
                   *     of the A and S blocks.
                   * @param Mppreconditioner Preconditioner object for the Schur complement,
                   *     typically chosen as the mass matrix.
                   * @param Apreconditioner Preconditioner object for the matrix A.
                   * @param do_solve_A A flag indicating whether we should actually solve with
                   *     the matrix $A$, or only apply one preconditioner step with it.
                   * @param A_block_tolerance The tolerance for the CG solver which computes
                   *     the inverse of the A block.
                   * @param S_block_tolerance The tolerance for the CG solver which computes
                   *     the inverse of the S block (Schur complement matrix).
                   **/
        BlockSchurPreconditioner (const StokesMatrixType  &S,
                                  const MassMatrixType  &Mass,
                                  const PreconditionerMp                     &Mppreconditioner,
                                  const PreconditionerA                      &Apreconditioner,
                                  const bool                                  do_solve_A,
                                  const double                                A_block_tolerance,
                                  const double                                S_block_tolerance,
                                  const MPI_Comm                         mpi_comm);

        /**
                   * Matrix vector product with this preconditioner object.
                   */
        void vmult (dealii::LinearAlgebra::distributed::BlockVector<double>       &dst,
                    const dealii::LinearAlgebra::distributed::BlockVector<double> &src) const;

        unsigned int n_iterations_A() const;
        unsigned int n_iterations_S() const;

        std::vector<double> return_times() const;



      private:
        /**
                   * References to the various matrix object this preconditioner works on.
                   */
        const StokesMatrixType &stokes_matrix;
        const MassMatrixType &mass_matrix;
        const PreconditionerMp                    &mp_preconditioner;
        const PreconditionerA                     &a_preconditioner;

        /**
                   * Whether to actually invert the $\tilde A$ part of the preconditioner matrix
                   * or to just apply a single preconditioner step with it.
                   **/
        const bool do_solve_A;
        mutable unsigned int n_iterations_A_;
        mutable unsigned int n_iterations_S_;
        const double A_block_tolerance;
        const double S_block_tolerance;

        mutable Timer time;
        mutable double a_prec_time;
        mutable double mass_solve_time;
        mutable double Bt_time;
    };


    template <class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
    BlockSchurPreconditioner<StokesMatrixType, MassMatrixType, PreconditionerMp, PreconditionerA>::
    BlockSchurPreconditioner (const StokesMatrixType  &S,
                              const MassMatrixType  &Mass,
                              const PreconditionerMp                     &Mppreconditioner,
                              const PreconditionerA                      &Apreconditioner,
                              const bool                                  do_solve_A,
                              const double                                A_block_tolerance,
                              const double                                S_block_tolerance,
                              const MPI_Comm                              mpi_comm)
      :
      stokes_matrix     (S),
      mass_matrix     (Mass),
      mp_preconditioner (Mppreconditioner),
      a_preconditioner  (Apreconditioner),
      do_solve_A        (do_solve_A),
      n_iterations_A_(0),
      n_iterations_S_(0),
      A_block_tolerance(A_block_tolerance),
      S_block_tolerance(S_block_tolerance),

      time(mpi_comm,true),
      a_prec_time(0.0), mass_solve_time(0.0), Bt_time(0.0)
    {}

    template <class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
    unsigned int
    BlockSchurPreconditioner<StokesMatrixType, MassMatrixType, PreconditionerMp, PreconditionerA>::
    n_iterations_A() const
    {
      return n_iterations_A_;
    }

    template <class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
    unsigned int
    BlockSchurPreconditioner<StokesMatrixType, MassMatrixType, PreconditionerMp, PreconditionerA>::
    n_iterations_S() const
    {
      return n_iterations_S_;
    }

    template <class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
    std::vector<double>
    BlockSchurPreconditioner<StokesMatrixType, MassMatrixType, PreconditionerMp, PreconditionerA>::
    return_times() const
    {
      std::vector<double> return_vec {a_prec_time, mass_solve_time, Bt_time};
      return return_vec;
    }

    template <class StokesMatrixType, class MassMatrixType, class PreconditionerMp,class PreconditionerA>
    void
    BlockSchurPreconditioner<StokesMatrixType, MassMatrixType, PreconditionerMp, PreconditionerA>::
    vmult (dealii::LinearAlgebra::distributed::BlockVector<double>       &dst,
           const dealii::LinearAlgebra::distributed::BlockVector<double>  &src) const
    {
      dealii::LinearAlgebra::distributed::BlockVector<double> utmp(src);

      // first solve with the bottom left block, which we have built
      // as a mass matrix with the inverse of the viscosity
      //time.restart();
      {
        SolverControl solver_control(1000, src.block(1).l2_norm() * S_block_tolerance,true);

        SolverCG<dealii::LinearAlgebra::distributed::Vector<double> > solver(solver_control);
        // Trilinos reports a breakdown
        // in case src=dst=0, even
        // though it should return
        // convergence without
        // iterating. We simply skip
        // solving in this case.
        if (src.block(1).l2_norm() > 1e-50)
          {
            try
              {
                dst.block(1) = 0.0;
                solver.solve(mass_matrix,
                             dst.block(1), src.block(1),
                             mp_preconditioner);
                n_iterations_S_ += solver_control.last_step();
              }
            // if the solver fails, report the error from processor 0 with some additional
            // information about its location, and throw a quiet exception on all other
            // processors
            catch (const std::exception &exc)
              {
                if (Utilities::MPI::this_mpi_process(src.block(0).get_mpi_communicator()) == 0)
                  AssertThrow (false,
                               ExcMessage (std::string("The iterative (bottom right) solver in BlockSchurPreconditioner::vmult "
                                                       "did not converge to a tolerance of "
                                                       + Utilities::to_string(solver_control.tolerance()) +
                                                       ". It reported the following error:\n\n")
                                           +
                                           exc.what()))
                  else
                    throw QuietException();
              }
          }
        dst.block(1) *= -1.0;
      }
      //time.stop();
      //mass_solve_time += time.last_wall_time();

      // apply the top right block
      //time.restart();
      {
        // TODO: figure out how to just multiply the top right block
        dealii::LinearAlgebra::distributed::BlockVector<double>  dst_tmp(dst);
        dst_tmp.block(0) = 0.0;
        stokes_matrix.vmult(utmp, dst_tmp); // B^T
        utmp.block(0) *= -1.0;
        utmp.block(0) += src.block(0);
      }
      //time.stop();
      //Bt_time += time.last_wall_time();

      // now either solve with the top left block (if do_solve_A==true)
      // or just apply one preconditioner sweep (for the first few
      // iterations of our two-stage outer GMRES iteration)
      if (do_solve_A == true)
        {
          Assert(false, ExcNotImplemented());
          /*
                        SolverControl solver_control(10000, utmp.block(0).l2_norm() * A_block_tolerance);
                        SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);
                        try
                        {
                          dst.block(0) = 0.0;
                          solver.solve(stokes_matrix.block(0,0), dst.block(0), utmp.block(0),
                                       a_preconditioner);
                          n_iterations_A_ += solver_control.last_step();
                        }
                        // if the solver fails, report the error from processor 0 with some additional
                        // information about its location, and throw a quiet exception on all other
                        // processors
                        catch (const std::exception &exc)
                        {
                          if (Utilities::MPI::this_mpi_process(src.block(0).get_mpi_communicator()) == 0)
                            AssertThrow (false,
                                         ExcMessage (std::string("The iterative (top left) solver in BlockSchurPreconditioner::vmult "
                                                                 "did not converge to a tolerance of "
                                                                 + Utilities::to_string(solver_control.tolerance()) +
                                                                 ". It reported the following error:\n\n")
                                                     +
                                                     exc.what()))
                                else
                                throw QuietException();
                        }
                        */
        }
      else
        {
          //time.restart();
          a_preconditioner.vmult (dst.block(0), utmp.block(0));
          n_iterations_A_ += 1;
          //time.stop();
          //a_prec_time += time.last_wall_time();
        }
    }
  }



  template <int dim>
  StokesMatrixFreeHandler<dim>::StokesMatrixFreeHandler (Simulator<dim> &simulator,
                                                         ParameterHandler &prm)
    : sim(simulator),

      stokes_fe (FE_Q<dim>(sim.parameters.stokes_velocity_degree),dim,
                 FE_Q<dim>(sim.parameters.stokes_velocity_degree-1),1),
      fe_v (FE_Q<dim>(sim.parameters.stokes_velocity_degree), dim),
      fe_p (FE_Q<dim>(sim.parameters.stokes_velocity_degree-1),1),

      dof_handler_v(sim.triangulation),
      dof_handler_p(sim.triangulation),

      dof_handler_projection(sim.triangulation),
      fe_projection(FE_DGQ<dim>(0),1)
  {
    parse_parameters(prm);
    //TODO CitationInfo::add("mf");

    // This requires: porting the additional stabilization terms and using a
    // different mapping in the MatrixFree operators:
    Assert(!sim.parameters.free_surface_enabled, ExcNotImplemented());
    // Sorry, not any time soon:
    Assert(!sim.parameters.include_melt_transport, ExcNotImplemented());
    // Not very difficult to do, but will require a different mass matrix
    // operator:
    Assert(!sim.parameters.use_locally_conservative_discretization, ExcNotImplemented());
    // TODO: this is currently hard-coded in the header:
    Assert(sim.parameters.stokes_velocity_degree==2, ExcNotImplemented());

    // sanity check:
    Assert(sim.introspection.variable("velocity").block_index==0, ExcNotImplemented());
    Assert(sim.introspection.variable("pressure").block_index==1, ExcNotImplemented());

    // This is not terribly complicated, but we need to check that constraints
    // are set correctly, that the preconditioner converges, and requires
    // testing.
    Assert(sim.geometry_model->get_periodic_boundary_pairs().size()==0, ExcNotImplemented());

    // We currently only support averaging that gives a constant value:
    using avg = MaterialModel::MaterialAveraging::AveragingOperation;
    Assert((sim.parameters.material_averaging &
            (avg::arithmetic_average | avg::harmonic_average | avg::geometric_average
             | avg::pick_largest | avg::log_average))!=0
           ,
           ExcNotImplemented());

    {
      const unsigned int n_vect_doubles =
        VectorizedArray<double>::n_array_elements;
      const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

      sim.pcout << "Vectorization over " << n_vect_doubles
                << " doubles = " << n_vect_bits << " bits ("
                << dealii::Utilities::System::get_current_vectorization_level()
                << "), VECTORIZATION_LEVEL=" << DEAL_II_COMPILER_VECTORIZATION_LEVEL
                << std::endl;
    }
  }

  template <int dim>
  StokesMatrixFreeHandler<dim>::~StokesMatrixFreeHandler ()
  {
  }

  template <int dim>
  void StokesMatrixFreeHandler<dim>::evaluate_viscosity ()
  {
    {
      const QGauss<dim> quadrature_formula (sim.parameters.stokes_velocity_degree+1);

      FEValues<dim> fe_values (*sim.mapping,
                               sim.finite_element,
                               quadrature_formula,
                               update_values   |
                               update_gradients |
                               update_quadrature_points |
                               update_JxW_values);

      MaterialModel::MaterialModelInputs<dim> in(fe_values.n_quadrature_points, sim.introspection.n_compositional_fields);
      MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points, sim.introspection.n_compositional_fields);

      std::vector<types::global_dof_index> local_dof_indices(fe_projection.dofs_per_cell);
      active_coef_dof_vec = 0.;

      // compute the integral quantities by quadrature
      for (const auto &cell: sim.dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            in.reinit(fe_values, cell, sim.introspection, sim.current_linearization_point);

            sim.material_model->fill_additional_material_model_inputs(in, sim.current_linearization_point, fe_values, sim.introspection);
            sim.material_model->evaluate(in, out);

            MaterialModel::MaterialAveraging::average (sim.parameters.material_averaging,
                                                       cell,
                                                       quadrature_formula,
                                                       *sim.mapping,
                                                       out);

            // we grab the first value, but all of them should be averaged to the same value:
            const double viscosity = out.viscosities[0];

            typename DoFHandler<dim>::active_cell_iterator dg_cell(&sim.triangulation,
                                                                   cell->level(),
                                                                   cell->index(),
                                                                   &dof_handler_projection);
            dg_cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < fe_projection.dofs_per_cell; ++i)
              active_coef_dof_vec[local_dof_indices[i]] = viscosity;
          }
      active_coef_dof_vec.compress(VectorOperation::insert);
    }

    // Fill viscosities table for active mesh
//      {
//        time.restart();

//        typename MatrixFree<dim,double>::AdditionalData additional_data;
//        additional_data.tasks_parallel_scheme =
//          MatrixFree<dim,double>::AdditionalData::none;
//        additional_data.mapping_update_flags = (update_values | update_quadrature_points);

//        std::vector<const DoFHandler<dim>*> dof_handlers;
//        dof_handlers.push_back(&dof_handler_v);
//        dof_handlers.push_back(&dof_handler_p);
//        dof_handlers.push_back(&dof_handler_projection);
//        std::vector<const ConstraintMatrix *> projection_constraints;
//        projection_constraints.push_back(&constraints_v);
//        projection_constraints.push_back(&constraints_p);
//        projection_constraints.push_back(&constraints_projection);

//        std::shared_ptr<MatrixFree<dim,double> >
//        mg_mf_storage(new MatrixFree<dim,double>());
//        mg_mf_storage->reinit(dof_handlers, projection_constraints,
//                              QGauss<1>(sim.parameters.stokes_velocity_degree+1), additional_data);

//        MatrixFreeStokesOperators::ComputeCoefficientProjection<dim,3,2,double> coeff_operator;
//        coeff_operator.initialize(mg_mf_storage);

//        dealii::LinearAlgebra::distributed::BlockVector<double> block_viscosity_values_stokes(3);
//        block_viscosity_values_stokes.collect_sizes();
//        coeff_operator.initialize_dof_vector(block_viscosity_values_stokes);

//        std::vector<types::global_dof_index> local_dof_indices(fe_projection.dofs_per_cell);

//        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_projection.begin_active(),
//                                                       endc = dof_handler_projection.end();
//        for (; cell != endc; ++cell)
//          if (cell->is_locally_owned())
//            {
//              cell->get_dof_indices(local_dof_indices);
//              for (unsigned int i=0; i<fe_projection.dofs_per_cell; ++i )
//                block_viscosity_values_stokes.block(2)(local_dof_indices[i]) = 2.0*active_coef_dof_vec(local_dof_indices[i]);
//            }
//        block_viscosity_values_stokes.compress(VectorOperation::insert);

//        stokes_matrix.fill_viscosities_and_pressure_scaling(coeff_operator.return_viscosity_table(block_viscosity_values_stokes.block(2)),
//                                                            sim.pressure_scaling);
//      }


    stokes_matrix.fill_viscosities_and_pressure_scaling(active_coef_dof_vec,
                                                        sim.pressure_scaling,
                                                        sim.triangulation,
                                                        dof_handler_projection);

//    if (sim.dof_handler.n_dofs() > 5000)
//      {
//        stokes_matrix.output_visc(Utilities::MPI::this_mpi_process(sim.triangulation.get_communicator()));
//        Assert(false,ExcInternalError());
//      }

//    {
//      typename MatrixFree<dim,double>::AdditionalData additional_data;
//      additional_data.tasks_parallel_scheme =
//        MatrixFree<dim,double>::AdditionalData::none;
//      additional_data.mapping_update_flags = (update_values | update_quadrature_points);

//      std::vector<const DoFHandler<dim>*> dof_handlers;
//      dof_handlers.push_back(&dof_handler_p);
//      dof_handlers.push_back(&dof_handler_projection);
//      std::vector<const ConstraintMatrix *> projection_constraints;
//      projection_constraints.push_back(&constraints_p);
//      projection_constraints.push_back(&constraints_projection);

//      std::shared_ptr<MatrixFree<dim,double> >
//      mg_mf_storage(new MatrixFree<dim,double>());
//      mg_mf_storage->reinit(dof_handlers, projection_constraints,
//                            QGauss<1>(sim.parameters.stokes_velocity_degree+1), additional_data);

//      MatrixFreeStokesOperators::ComputeCoefficientProjection<dim,3,1,double> coeff_operator;
//      coeff_operator.initialize(mg_mf_storage);

//      dealii::LinearAlgebra::distributed::BlockVector<double> block_viscosity_values_mass(2);
//      block_viscosity_values_mass.collect_sizes();
//      coeff_operator.initialize_dof_vector(block_viscosity_values_mass);

//      std::vector<types::global_dof_index> local_dof_indices(fe_projection.dofs_per_cell);

//      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_projection.begin_active(),
//                                                     endc = dof_handler_projection.end();
//      for (; cell != endc; ++cell)
//        if (cell->is_locally_owned())
//          {
//            cell->get_dof_indices(local_dof_indices);
//            for (unsigned int i=0; i<fe_projection.dofs_per_cell; ++i )
//              {
//                block_viscosity_values_mass.block(1)(local_dof_indices[i]) = 1.0/active_coef_dof_vec(local_dof_indices[i]);
//              }
//          }
//      block_viscosity_values_mass.compress(VectorOperation::insert);

//      mass_matrix.fill_viscosities_and_pressure_scaling(coeff_operator.return_viscosity_table(block_viscosity_values_mass.block(1)),
//                                                        sim.pressure_scaling);

//      //      if (sim.dof_handler.n_dofs() > 5000)
//      //      {
//      //          stokes_matrix.output_visc(Utilities::MPI::this_mpi_process(sim.triangulation.get_communicator()));
//      //          mass_matrix.output_visc(Utilities::MPI::this_mpi_process(sim.triangulation.get_communicator()));
//      //          //Assert(false,ExcInternalError());
//      //      }

//      mass_matrix.compute_diagonal();
//    }

    mass_matrix.fill_viscosities_and_pressure_scaling(active_coef_dof_vec,
                                                      sim.pressure_scaling,
                                                      sim.triangulation,
                                                      dof_handler_projection);
    mass_matrix.compute_diagonal();


    // Fill viscosities for the level operators
//    {
//      unsigned int n_levels = sim.triangulation.n_global_levels();
//      level_coef_dof_vec = 0.;
//      level_coef_dof_vec.resize(0,n_levels-1);

//      MGTransferMatrixFree<dim,double> transfer(mg_constrained_dofs);
//      transfer.build(dof_handler_projection);
//      transfer.interpolate_to_mg(dof_handler_projection,level_coef_dof_vec,active_coef_dof_vec);

//      bool should_i_break = false;
//      for (unsigned int level=0; level<n_levels; ++level)
//        {
//          ConstraintMatrix level_constraints;
//          {
//            IndexSet relevant_dofs;
//            DoFTools::extract_locally_relevant_level_dofs(dof_handler_v, level, relevant_dofs);
//            level_constraints.reinit(relevant_dofs);
//            level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
//            level_constraints.close();
//          }

//          ConstraintMatrix level_constraints_projection;
//          {
//            IndexSet relevant_dofs;
//            DoFTools::extract_locally_relevant_level_dofs(dof_handler_projection, level, relevant_dofs);
//            level_constraints_projection.reinit(relevant_dofs);
//            level_constraints_projection.close();
//          }

//          typename MatrixFree<dim,double>::AdditionalData additional_data;
//          additional_data.tasks_parallel_scheme =
//            MatrixFree<dim,double>::AdditionalData::none;
//          additional_data.mapping_update_flags = (update_values | update_quadrature_points);
//          additional_data.level_mg_handler = level;

//          std::vector<const DoFHandler<dim>*> dof_handlers;
//          dof_handlers.push_back(&dof_handler_v);
//          dof_handlers.push_back(&dof_handler_projection);
//          std::vector<const ConstraintMatrix *> projection_constraints;
//          projection_constraints.push_back(&level_constraints);
//          projection_constraints.push_back(&level_constraints_projection);

//          std::shared_ptr<MatrixFree<dim,double> >
//          mg_mf_storage_level(new MatrixFree<dim,double>());
//          mg_mf_storage_level->reinit(dof_handlers, projection_constraints,
//                                      QGauss<1>(sim.parameters.stokes_velocity_degree+1),
//                                      additional_data);

//          std::vector<MGConstrainedDoFs> mg_constrained_dofs_vec;
//          mg_constrained_dofs_vec.push_back(mg_constrained_dofs);
//          mg_constrained_dofs_vec.push_back(mg_constrained_dofs_projection);

//          MatrixFreeStokesOperators::ComputeCoefficientProjection<dim,3,1,double> coeff_operator;
//          coeff_operator.initialize(mg_mf_storage_level, mg_constrained_dofs_vec, level);

//          dealii::LinearAlgebra::distributed::BlockVector<double> block_viscosity_values_level(2);
//          block_viscosity_values_level.collect_sizes();
//          coeff_operator.initialize_dof_vector(block_viscosity_values_level);

//          std::vector<types::global_dof_index> local_dof_indices(fe_projection.dofs_per_cell);
//          typename DoFHandler<dim>::cell_iterator cell = dof_handler_projection.begin(level),
//                                                  endc = dof_handler_projection.end(level);
//          for (; cell != endc; ++cell)
//            if (cell->is_locally_owned_on_level())
//              {
//                cell->get_mg_dof_indices(local_dof_indices);
//                for (unsigned int i=0; i<fe_projection.dofs_per_cell; ++i )
//                  block_viscosity_values_level.block(1)(local_dof_indices[i]) = 2.0*level_coef_dof_vec[level](local_dof_indices[i]);
//              }
//          block_viscosity_values_level.compress(VectorOperation::insert);

//          mg_matrices[level].fill_viscosities(coeff_operator.return_viscosity_table(block_viscosity_values_level.block(1)));
//          if (sim.dof_handler.n_dofs() > 10000)
//            {
//              GridOut grid_out;
//              grid_out.write_mesh_per_processor_as_vtu(sim.triangulation,"grid",true,false);
//              mg_matrices[level].output_visc(level,Utilities::MPI::this_mpi_process(sim.mpi_communicator));
//              should_i_break = true;
//            }
//          mg_matrices[level].compute_diagonal();
//        }
//      if (should_i_break)
//        Assert(false,ExcNotImplemented());
//    }

    const unsigned int n_levels = sim.triangulation.n_global_levels();
    level_coef_dof_vec = 0.;
    level_coef_dof_vec.resize(0,n_levels-1);

    MGTransferMatrixFree<dim,double> transfer(mg_constrained_dofs);
    transfer.build(dof_handler_projection);
    transfer.interpolate_to_mg(dof_handler_projection,
                               level_coef_dof_vec,
                               active_coef_dof_vec);

    for (unsigned int level=0; level<n_levels; ++level)
      {
        mg_matrices[level].fill_viscosities(level_coef_dof_vec,
                                            sim.triangulation,
                                            dof_handler_projection);
        mg_matrices[level].compute_diagonal();
      }
  }


  template <int dim>
  void StokesMatrixFreeHandler<dim>::correct_stokes_rhs()
  {
    dealii::LinearAlgebra::distributed::BlockVector<double> rhs_correction(2);
    dealii::LinearAlgebra::distributed::BlockVector<double> u0(2);
    rhs_correction.collect_sizes();
    u0.collect_sizes();
    stokes_matrix.initialize_dof_vector(rhs_correction);
    stokes_matrix.initialize_dof_vector(u0);

    u0 = 0;
    rhs_correction = 0;
    sim.current_constraints.distribute(u0);
    u0.update_ghost_values();

    const Table<2, VectorizedArray<double>> viscosity_table = stokes_matrix.get_visc_table();
    FEEvaluation<dim,2,3,dim,double>
    velocity (*stokes_matrix.get_matrix_free(), 0);
    FEEvaluation<dim,1,3,1,double>
    pressure (*stokes_matrix.get_matrix_free(), 1);

    for (unsigned int cell=0; cell<stokes_matrix.get_matrix_free()->n_macro_cells(); ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values_plain (u0.block(0));
        velocity.evaluate (false,true,false);
        pressure.reinit (cell);
        pressure.read_dof_values_plain (u0.block(1));
        pressure.evaluate (true,false,false);

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            SymmetricTensor<2,dim,VectorizedArray<double>> sym_grad_u =
                                                          velocity.get_symmetric_gradient (q);
            VectorizedArray<double> pres = pressure.get_value(q);
            VectorizedArray<double> div = -trace(sym_grad_u);
            pressure.submit_value   (-1.0*sim.pressure_scaling*div, q);

            sym_grad_u *= viscosity_table(cell,q);

            for (unsigned int d=0; d<dim; ++d)
              sym_grad_u[d][d] -= sim.pressure_scaling*pres;

            velocity.submit_symmetric_gradient(-1.0*sym_grad_u, q);
          }

        velocity.integrate (false,true);
        velocity.distribute_local_to_global (rhs_correction.block(0));
        pressure.integrate (true,false);
        pressure.distribute_local_to_global (rhs_correction.block(1));
      }
    rhs_correction.compress(VectorOperation::add);

    LinearAlgebra::BlockVector stokes_rhs_correction (sim.introspection.index_sets.stokes_partitioning, sim.mpi_communicator);
    internal::ChangeVectorTypes::copy(stokes_rhs_correction,rhs_correction);
    sim.system_rhs.block(0) += stokes_rhs_correction.block(0);
    sim.system_rhs.block(1) += stokes_rhs_correction.block(1);
  }


  template <int dim>
  void StokesMatrixFreeHandler<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection ("Solver parameters");
    prm.enter_subsection ("Matrix Free");
    {
      prm.declare_entry("Free surface stabilization theta", "0.5",
                        Patterns::Double(0,1),
                        "Theta parameter described in Kaus et. al. 2010. "
                        "An unstabilized free surface can overshoot its "
                        "equilibrium position quite easily and generate "
                        "unphysical results.  One solution is to use a "
                        "quasi-implicit correction term to the forces near the "
                        "free surface.  This parameter describes how much "
                        "the free surface is stabilized with this term, "
                        "where zero is no stabilization, and one is fully "
                        "implicit.");
    }
    prm.leave_subsection ();
    prm.leave_subsection ();
  }

  template <int dim>
  void StokesMatrixFreeHandler<dim>::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection ("Solver parameters");
    prm.enter_subsection ("Matrix Free");
    {
      //free_surface_theta = prm.get_double("Free surface stabilization theta");
    }
    prm.leave_subsection ();
    prm.leave_subsection ();
  }



  template <int dim>
  std::pair<double,double> StokesMatrixFreeHandler<dim>::solve(unsigned int i)
  {
    sim.pcout << "solve() "
              << std::endl;

    //Update coefficients and add correction to system rhs
    evaluate_viscosity();
    correct_stokes_rhs();


    double initial_nonlinear_residual = numbers::signaling_nan<double>();
    double final_linear_residual      = numbers::signaling_nan<double>();

    LinearAlgebra::BlockVector distributed_stokes_solution (sim.introspection.index_sets.stokes_partitioning, sim.mpi_communicator);


    typedef dealii::LinearAlgebra::distributed::Vector<double> vector_t;


    //    //Jacobi Smoother
    //        typedef PreconditionJacobi<ABlockMatrixType> SmootherType;
    //        MGSmootherPrecondition<ABlockMatrixType, SmootherType, vector_t> mg_smoother;
    //        mg_smoother.initialize(mg_matrices, typename SmootherType::AdditionalData(0.4));
    //        mg_smoother.set_steps(4);

    //Chebyshev smoother
    typedef PreconditionChebyshev<ABlockMatrixType,vector_t> SmootherType;
    mg::SmootherRelaxation<SmootherType, vector_t>
    mg_smoother;
    {
      MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
      smoother_data.resize(0, sim.triangulation.n_global_levels()-1);
      for (unsigned int level = 0; level<sim.triangulation.n_global_levels(); ++level)
        {
          if (level > 0)
            {
              smoother_data[level].smoothing_range = 15.;
              smoother_data[level].degree = 4;
              smoother_data[level].eig_cg_n_iterations = 10;
            }
          else
            {
              smoother_data[0].smoothing_range = 1e-3;
              smoother_data[0].degree = numbers::invalid_unsigned_int;
              smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
            }
          smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
        }
      mg_smoother.initialize(mg_matrices, smoother_data);
    }


    // Coarse Solver
    // TODO: This could help with tangential boundary
    MGCoarseGridApplySmoother<vector_t> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    // Interface matrices
    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<ABlockMatrixType> > mg_interface_matrices;
    mg_interface_matrices.resize(0, sim.triangulation.n_global_levels()-1);
    for (unsigned int level=0; level<sim.triangulation.n_global_levels(); ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<vector_t > mg_interface(mg_interface_matrices);

    // MG Matrix
    mg::Matrix<vector_t > mg_matrix(mg_matrices);

    // MG object
    Multigrid<vector_t > mg(mg_matrix,
                            mg_coarse,
                            mg_transfer,
                            mg_smoother,
                            mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    //mg.set_debug(10);


    // MG Preconditioner
    typedef PreconditionMG<dim, vector_t, MGTransferMatrixFree<dim,double> > APreconditioner;
    APreconditioner prec_A(dof_handler_v, mg, mg_transfer);

    //Mass matrix Preconditioner
    typedef PreconditionChebyshev<MassMatrixType,vector_t> MassPreconditioner;
    MassPreconditioner prec_S;
    typename MassPreconditioner::AdditionalData prec_S_data;
    prec_S_data.smoothing_range = 1e-3;
    prec_S_data.degree = numbers::invalid_unsigned_int;
    prec_S_data.eig_cg_n_iterations = mass_matrix.m();
    prec_S_data.preconditioner = mass_matrix.get_matrix_diagonal_inverse();
    prec_S.initialize(mass_matrix,prec_S_data);















    // Many parts of the solver depend on the block layout (velocity = 0,
    // pressure = 1). For example the linearized_stokes_initial_guess vector or the StokesBlock matrix
    // wrapper. Let us make sure that this holds (and shorten their names):
    const unsigned int block_vel = sim.introspection.block_indices.velocities;
    const unsigned int block_p = (sim.parameters.include_melt_transport) ?
                                 sim.introspection.variable("fluid pressure").block_index
                                 : sim.introspection.block_indices.pressure;
    Assert(block_vel == 0, ExcNotImplemented());
    Assert(block_p == 1, ExcNotImplemented());
    Assert(!sim.parameters.include_melt_transport
           || sim.introspection.variable("compaction pressure").block_index == 1, ExcNotImplemented());


    // create a completely distributed vector that will be used for
    // the scaled and denormalized solution and later used as a
    // starting guess for the linear solver
    LinearAlgebra::BlockVector linearized_stokes_initial_guess (sim.introspection.index_sets.stokes_partitioning, sim.mpi_communicator);

    // copy the velocity and pressure from current_linearization_point into
    // the vector linearized_stokes_initial_guess. We need to do the copy because
    // linearized_stokes_variables has a different
    // layout than current_linearization_point, which also contains all the
    // other solution variables.
    if (!sim.assemble_newton_stokes_system)
      {
        linearized_stokes_initial_guess.block (block_vel) = sim.current_linearization_point.block (block_vel);
        linearized_stokes_initial_guess.block (block_p) = sim.current_linearization_point.block (block_p);

        sim.denormalize_pressure (sim.last_pressure_normalization_adjustment,
                                  linearized_stokes_initial_guess,
                                  sim.current_linearization_point);
      }
    else
      {
        // The Newton solver solves for updates to variables, for which our best guess is zero when
        // the it isn't the first nonlinear iteration. When it is the first nonlinear iteration, we
        // have to assemble the full (non-defect correction) Picard, to get the boundary conditions
        // right in combination with being able to use the initial guess optimally. So we may never
        // end up here when it is the first nonlinear iteration.
        Assert(sim.nonlinear_iteration != 0,
               ExcMessage ("The Newton solver may not be active in the first nonlinear iteration"));

        linearized_stokes_initial_guess.block (block_vel) = 0;
        linearized_stokes_initial_guess.block (block_p) = 0;
      }

    sim.current_constraints.set_zero (linearized_stokes_initial_guess);
    linearized_stokes_initial_guess.block (block_p) /= sim.pressure_scaling;


    double solver_tolerance = 0;
    if (sim.assemble_newton_stokes_system == false)
      {
        // (ab)use the distributed solution vector to temporarily put a residual in
        // (we don't care about the residual vector -- all we care about is the
        // value (number) of the initial residual). The initial residual is returned
        // to the caller (for nonlinear computations). This value is computed before
        // the solve because we want to compute || A^{k+1} U^k - F^{k+1} ||, which is
        // the nonlinear residual. Because the place where the nonlinear residual is
        // checked against the nonlinear tolerance comes after the solve, the system
        // is solved one time too many in the case of a nonlinear Picard solver.
        LinearAlgebra::BlockVector rhs_tmp (sim.introspection.index_sets.stokes_partitioning, sim.mpi_communicator);
        rhs_tmp.block(0) = sim.system_rhs.block(0);
        rhs_tmp.block(1) = sim.system_rhs.block(1);

        dealii::LinearAlgebra::distributed::BlockVector<double> x(2);
        x.collect_sizes();
        stokes_matrix.initialize_dof_vector(x);
        internal::ChangeVectorTypes::copy(x,linearized_stokes_initial_guess);

        dealii::LinearAlgebra::distributed::BlockVector<double> Ax(2);
        Ax.collect_sizes();
        stokes_matrix.initialize_dof_vector(Ax);
        stokes_matrix.vmult(Ax,x);

        LinearAlgebra::BlockVector res (sim.introspection.index_sets.stokes_partitioning, sim.mpi_communicator);
        internal::ChangeVectorTypes::copy(res,Ax);
        res *= -1;
        res += rhs_tmp;
        initial_nonlinear_residual = res.l2_norm();

        // Note: the residual is computed with a zero velocity, effectively computing
        // || B^T p - g ||, which we are going to use for our solver tolerance.
        // We do not use the current velocity for the initial residual because
        // this would not decrease the number of iterations if we had a better
        // initial guess (say using a smaller timestep). But we need to use
        // the pressure instead of only using the norm of the rhs, because we
        // are only interested in the part of the rhs not balanced by the static
        // pressure (the current pressure is a good approximation for the static
        // pressure).
        x.block(0) = 0;
        dealii::LinearAlgebra::distributed::BlockVector<double> Btp(2);
        Btp.collect_sizes();
        stokes_matrix.initialize_dof_vector(Btp);
        stokes_matrix.vmult(Btp,x);

        res = 0;
        internal::ChangeVectorTypes::copy(res,Btp);
        res *= -1;
        res += rhs_tmp;
        const double residual_u = res.block(0).l2_norm();

        const double residual_p = sim.system_rhs.block(1).l2_norm();

        solver_tolerance = sim.parameters.linear_stokes_solver_tolerance *
                           std::sqrt(residual_u*residual_u+residual_p*residual_p);
      }
    else
      {
        // if we are solving for the Newton update, then the initial guess of the solution
        // vector is the zero vector, and the starting (nonlinear) residual is simply
        // the norm of the (Newton) right hand side vector
        const double residual_u = sim.system_rhs.block(0).l2_norm();
        const double residual_p = sim.system_rhs.block(1).l2_norm();
        solver_tolerance = sim.parameters.linear_stokes_solver_tolerance *
                           std::sqrt(residual_u*residual_u+residual_p*residual_p);

        // as described in the documentation of the function, the initial
        // nonlinear residual for the Newton method is computed by just
        // taking the norm of the right hand side
        initial_nonlinear_residual = std::sqrt(residual_u*residual_u+residual_p*residual_p);
      }
    // Now overwrite the solution vector again with the current best guess
    // to solve the linear system
    distributed_stokes_solution = linearized_stokes_initial_guess;

    // extract Stokes parts of rhs vector
    LinearAlgebra::BlockVector distributed_stokes_rhs(sim.introspection.index_sets.stokes_partitioning);

    distributed_stokes_rhs.block(block_vel) = sim.system_rhs.block(block_vel);
    distributed_stokes_rhs.block(block_p) = sim.system_rhs.block(block_p);

    PrimitiveVectorMemory<dealii::LinearAlgebra::distributed::BlockVector<double> > mem;


    // create Solver controls for the cheap and expensive solver phase
    SolverControl solver_control_cheap (sim.parameters.n_cheap_stokes_solver_steps,
                                        solver_tolerance, true);
    //  SolverControl solver_control_expensive (sim.parameters.n_expensive_stokes_solver_steps,
    //                                          solver_tolerance);

    solver_control_cheap.enable_history_data();
    //solver_control_expensive.enable_history_data();


    // create a cheap preconditioner that consists of only a single V-cycle
    const internal::BlockSchurPreconditioner<StokesMatrixType, MassMatrixType, MassPreconditioner, APreconditioner>
    preconditioner_cheap (stokes_matrix, mass_matrix,
                          prec_S, prec_A,
                          false,
                          sim.parameters.linear_solver_A_block_tolerance,
                          sim.parameters.linear_solver_S_block_tolerance,
                          sim.triangulation.get_communicator());

    dealii::LinearAlgebra::distributed::BlockVector<double> solution_copy(2);
    dealii::LinearAlgebra::distributed::BlockVector<double> rhs_copy(2);
    solution_copy.collect_sizes();
    rhs_copy.collect_sizes();
    stokes_matrix.initialize_dof_vector(solution_copy);
    stokes_matrix.initialize_dof_vector(rhs_copy);
    internal::ChangeVectorTypes::copy(solution_copy,distributed_stokes_solution);
    internal::ChangeVectorTypes::copy(rhs_copy,distributed_stokes_rhs);
    solution_copy.update_ghost_values();


    if (sim.parameters.n_timings==0)
      sim.vcycle_time =0;
    else if (i==0)
      {
        sim.vcycle_time = 0;
        dealii::LinearAlgebra::distributed::BlockVector<double> tmp_dst = solution_copy;
        dealii::LinearAlgebra::distributed::BlockVector<double> tmp_scr = rhs_copy;
        Timer time(sim.triangulation.get_communicator(),true);
        for (unsigned int i=0; i<sim.parameters.n_timings+1; ++i)
          {
            time.restart();

            preconditioner_cheap.vmult(tmp_dst, tmp_scr);

            time.stop();
            if (i!=0)
              sim.vcycle_time += time.last_wall_time();

            tmp_scr = tmp_dst;
          }
        if (sim.parameters.n_timings != 0)
          sim.vcycle_time /= (1.0*sim.parameters.n_timings);
      }


    // TODO: doesn't make a difference
    //solution_copy = 0.;

//    deallog.depth_console(10);
//    std::ofstream solve_data("/hdscratch/tcleven/aspect-git/aspect/benchmarks/nsinker/solve_data.log");
//    deallog.attach(solve_data);


    sim.gmres_iterations = 0;
    try
      {
        SolverFGMRES<dealii::LinearAlgebra::distributed::BlockVector<double> >
        solver(solver_control_cheap, mem,
               SolverFGMRES<dealii::LinearAlgebra::distributed::BlockVector<double> >::
               AdditionalData(50));

        solver.solve (stokes_matrix,
                      solution_copy,
                      rhs_copy,
                      preconditioner_cheap);

        final_linear_residual = solver_control_cheap.last_value();
      }
    catch (SolverControl::NoConvergence)
      {
        sim.pcout << "********************************************************************" << std::endl
                  << "SOLVER DID NOT CONVERGE AFTER "
                  << sim.parameters.n_cheap_stokes_solver_steps
                  << " ITERATIONS. res=" << solver_control_cheap.last_value() << std::endl
                  << "********************************************************************" << std::endl;

        //Assert(false,ExcNotImplemented());
      }
    sim.gmres_iterations = solver_control_cheap.last_step();



    // signal successful solver
    //  signals.post_stokes_solver(*this,
    //                             preconditioner_cheap.n_iterations_S() + preconditioner_expensive.n_iterations_S(),
    //                             preconditioner_cheap.n_iterations_A() + preconditioner_expensive.n_iterations_A(),
    //                             solver_control_cheap,
    //                             solver_control_expensive);

    // distribute hanging node and
    // other constraints
    solution_copy.update_ghost_values();
    internal::ChangeVectorTypes::copy(distributed_stokes_solution,solution_copy);

    sim.current_constraints.distribute (distributed_stokes_solution);

    // now rescale the pressure back to real physical units
    distributed_stokes_solution.block(block_p) *= sim.pressure_scaling;

    // then copy back the solution from the temporary (non-ghosted) vector
    // into the ghosted one with all solution components
    sim.solution.block(block_vel) = distributed_stokes_solution.block(block_vel);
    sim.solution.block(block_p) = distributed_stokes_solution.block(block_p);

    // print the number of iterations to screen
    if (i==0)
      {
        sim.pcout << std::left
                  << std::setw(8) << "out:"
                  << "GMRES iterations "
                  << std::endl
                  << std::setw(8) << "out:"
                  << (solver_control_cheap.last_step() != numbers::invalid_unsigned_int ?
                      solver_control_cheap.last_step():
                      0)
                  << std::endl;
        sim.pcout << std::left
                  << std::setw(8) << "out:" << std::endl;
      }


    // do some cleanup now that we have the solution
    sim.remove_nullspace(sim.solution, distributed_stokes_solution);
    if (!sim.assemble_newton_stokes_system)
      sim.last_pressure_normalization_adjustment = sim.normalize_pressure(sim.solution);

    // convert melt pressures:
    //if (sim.parameters.include_melt_transport)
    //melt_handler->compute_melt_variables(solution);

    return std::pair<double,double>(initial_nonlinear_residual,
                                    final_linear_residual);
  }



  template <int dim>
  void StokesMatrixFreeHandler<dim>::setup_dofs()
  {

//    Timer time(sim.triangulation.get_communicator(),true);
//    double dof_setup = 0.0, mf_setup = 0.0, gmg_transfer = 0.0;

//    time.restart();
    {
      dof_handler_v.clear();
      dof_handler_v.distribute_dofs(fe_v);
      dof_handler_v.distribute_mg_dofs();

      DoFRenumbering::hierarchical(dof_handler_v);

      constraints_v.clear();
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs (dof_handler_v,
                                               locally_relevant_dofs);
      constraints_v.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints (dof_handler_v, constraints_v);
      sim.compute_initial_velocity_boundary_constraints(constraints_v);
      sim.compute_current_velocity_boundary_constraints(constraints_v);
      constraints_v.close ();
    }

    {
      dof_handler_p.clear();
      dof_handler_p.distribute_dofs(fe_p);

      DoFRenumbering::hierarchical(dof_handler_p);

      constraints_p.clear();
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs (dof_handler_p,
                                               locally_relevant_dofs);
      constraints_p.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints (dof_handler_p, constraints_p);
      constraints_p.close();
    }

    // Coefficient transfer objects
    {
      dof_handler_projection.clear();
      dof_handler_projection.distribute_dofs(fe_projection);
      dof_handler_projection.distribute_mg_dofs();

      DoFRenumbering::hierarchical(dof_handler_projection);


      constraints_projection.clear();
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs (dof_handler_projection,
                                               locally_relevant_dofs);
      constraints_projection.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints (dof_handler_projection, constraints_projection);
      // TODO: boundary constraints if not DG(0)?
      constraints_projection.close ();

      mg_constrained_dofs_projection.clear();
      mg_constrained_dofs_projection.initialize(dof_handler_projection);

      active_coef_dof_vec.reinit(dof_handler_projection.locally_owned_dofs(), sim.triangulation.get_communicator());
    }
//    time.stop();
//    dof_setup += time.last_wall_time();



//    time.restart();
    // Stokes matrix stuff...
    {
      typename MatrixFree<dim,double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim,double>::AdditionalData::none;
      additional_data.mapping_update_flags = (update_values | update_gradients |
                                              update_JxW_values | update_quadrature_points);

      std::vector<const DoFHandler<dim>*> stokes_dofs;
      stokes_dofs.push_back(&dof_handler_v);
      stokes_dofs.push_back(&dof_handler_p);
      std::vector<const ConstraintMatrix *> stokes_constraints;
      stokes_constraints.push_back(&constraints_v);
      stokes_constraints.push_back(&constraints_p);

      std::shared_ptr<MatrixFree<dim,double> >
      stokes_mf_storage(new MatrixFree<dim,double>());
      stokes_mf_storage->reinit(stokes_dofs, stokes_constraints,
                                QGauss<1>(sim.parameters.stokes_velocity_degree+1), additional_data);
      stokes_matrix.clear();
      stokes_matrix.initialize(stokes_mf_storage);
    }

    // Mass matrix matrix-free operator...
    {
      typename MatrixFree<dim,double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim,double>::AdditionalData::none;
      additional_data.mapping_update_flags = (update_values | update_JxW_values |
                                              update_quadrature_points);
      std::shared_ptr<MatrixFree<dim,double> >
      mass_mf_storage(new MatrixFree<dim,double>());
      mass_mf_storage->reinit(dof_handler_p, constraints_p,
                              QGauss<1>(sim.parameters.stokes_velocity_degree+1), additional_data);

      mass_matrix.clear();
      mass_matrix.initialize(mass_mf_storage);
    }

    // GMG Matrix-free operators
    {
      const unsigned int n_levels = sim.triangulation.n_global_levels();
      mg_matrices.clear_elements();
      // TODO: minlevel != 0
      mg_matrices.resize(0, n_levels-1);

      mg_constrained_dofs.clear();
      std::set<types::boundary_id> dirichlet_boundary = sim.boundary_velocity_manager.get_zero_boundary_velocity_indicators();
      //TODO: Assert no tangetial boundary
      for (auto it: sim.boundary_velocity_manager.get_active_boundary_velocity_names())
        {
          int bdryid = it.first;
          std::string component=it.second.first;
          Assert(component=="", ExcNotImplemented());
          dirichlet_boundary.insert(bdryid);
        }

      mg_constrained_dofs.initialize(dof_handler_v);
      mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_v, dirichlet_boundary);

      for (unsigned int level=0; level<n_levels; ++level)
        {
          IndexSet relevant_dofs;
          DoFTools::extract_locally_relevant_level_dofs(dof_handler_v, level, relevant_dofs);
          ConstraintMatrix level_constraints;
          level_constraints.reinit(relevant_dofs);
          level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
          level_constraints.close();

          {
            typename MatrixFree<dim,double>::AdditionalData additional_data;
            additional_data.tasks_parallel_scheme =
              MatrixFree<dim,double>::AdditionalData::none;
            additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                                    update_quadrature_points);
            additional_data.level_mg_handler = level;
            std::shared_ptr<MatrixFree<dim,double> >
            mg_mf_storage_level(new MatrixFree<dim,double>());
            mg_mf_storage_level->reinit(dof_handler_v, level_constraints,
                                        QGauss<1>(sim.parameters.stokes_velocity_degree+1),
                                        additional_data);

            mg_matrices[level].clear();
            mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs, level);

            //add zero boundary coarsest level
          }
        }
    }
//    time.stop();
//    mf_setup += time.last_wall_time();



    //Setup GMG transfer
//    time.restart();
    {
      mg_transfer.clear();
      mg_transfer.initialize_constraints(mg_constrained_dofs);
      mg_transfer.build(dof_handler_v);
    }
//    time.stop();
//    gmg_transfer += time.last_wall_time();

//    sim.pcout << std::left
//              << std::setw(8) << "out:"
//              << std::setw(15) << "MF-setup"
//              << std::setw(15) << "dofs"
//              << std::setw(15) << "mf"
//              << std::setw(15) << "gmg-transfer"
//              << std::endl;
//    sim.pcout << std::left
//              << std::setw(8) << "out:"
//              << std::setw(15) << dof_setup+mf_setup+gmg_transfer
//              << std::setw(15) << dof_setup
//              << std::setw(15) << mf_setup
//              << std::setw(15) << gmg_transfer
//              << std::endl
//              << std::setw(8) << "out:" << std::endl;



  }

}



// explicit instantiation of the functions we implement in this file
namespace aspect
{
#define INSTANTIATE(dim) \
  template class StokesMatrixFreeHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)
}
