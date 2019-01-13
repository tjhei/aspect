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


#include <aspect/simulator.h>
#include <aspect/stokes_matrix_free.h>
#include <aspect/global.h>
#include <aspect/citation_info.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>


#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace internal
  {

  }

  template <int dim>
  StokesMatrixFreeHandler<dim>::StokesMatrixFreeHandler (Simulator<dim> &simulator,
                                                         ParameterHandler &prm)
    : sim(simulator),

      fe_v (FE_Q<dim>(sim.parameters.stokes_velocity_degree), dim),
      fe_p (FE_Q<dim>(sim.parameters.stokes_velocity_degree-1),1),

      dof_handler_v(sim.triangulation),
      dof_handler_p(sim.triangulation)
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
  }

  template <int dim>
  StokesMatrixFreeHandler<dim>::~StokesMatrixFreeHandler ()
  {

  }

  template <int dim>
  void StokesMatrixFreeHandler<dim>::evaluate_viscosity ()
  {

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
  std::pair<double,double> StokesMatrixFreeHandler<dim>::solve()
  {
    sim.pcout << "solve() "
              << std::endl;


    double initial_nonlinear_residual = numbers::signaling_nan<double>();
    double final_linear_residual      = numbers::signaling_nan<double>();
    return std::pair<double,double>(initial_nonlinear_residual,
                                    final_linear_residual);
  }



  template <int dim>
  void StokesMatrixFreeHandler<dim>::setup_dofs()
  {
    sim.pcout << "Number of free surface degrees of freedom: "
              << std::endl;


    dof_handler_v.distribute_dofs(fe_v);
    dof_handler_v.distribute_mg_dofs();

    dof_handler_p.clear();
    dof_handler_p.distribute_dofs(fe_p);

    IndexSet locally_relevant_dofs_v;
    DoFTools::extract_locally_relevant_dofs (dof_handler_v,
                                             locally_relevant_dofs_v);
    constraints_v.reinit(locally_relevant_dofs_v);
    DoFTools::make_hanging_node_constraints (dof_handler_v, constraints_v);
    sim.compute_initial_velocity_boundary_constraints(constraints_v);
    sim.compute_current_velocity_boundary_constraints(constraints_v);
    constraints_v.close ();

    IndexSet locally_relevant_dofs_p;
    DoFTools::extract_locally_relevant_dofs (dof_handler_p,
                                             locally_relevant_dofs_p);
    constraints_p.reinit(locally_relevant_dofs_p);
    DoFTools::make_hanging_node_constraints (dof_handler_p, constraints_p);
    constraints_p.close();


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
      stokes_matrix.initialize(stokes_mf_storage);

      // TODO: Get viscosity table
//      const unsigned int n_cells = stokes_mf_storage->n_macro_cells();
//      FEEvaluation<dim,sim.parameters.stokes_velocity_degree,
//                   sim.parameters.stokes_velocity_degree+1,
//                   dim,double> velocity (stokes_mf_storage, 0);
//      const unsigned int n_q_points = velocity.n_q_points;
//      const Table<2,VectorizedArray<double> > visc_vals;
//      visc_vals.reinit (n_cells, n_q_points);
//      stokes_matrix.evaluate_2_x_viscosity(visc_vals);
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
                              QGauss<1>(velocity_degree+1), additional_data);

      mass_matrix.initialize(mass_mf_storage);

      // TODO: Get viscosity table/pressure scaling
//      const unsigned int n_cells = mass_mf_storage->n_macro_cells();
//      FEEvaluation<dim,sim.parameters.stokes_velocity_degree-1,
//                   sim.parameters.stokes_velocity_degree+1,
//                   dim,double> pressure (mass_mf_storage, 0);
//      const unsigned int n_q_points = pressure.n_q_points;
//      const Table<2,VectorizedArray<double> > visc_vals;
//      visc_vals.reinit (n_cells, n_q_points);
//      mass_matrix.evaluate_1_over_viscosity_and_scaling(visc_vals, 1.0);
//      mass_matrix.compute_diagonal();
    }
  }


}


// explicit instantiation of the functions we implement in this file
namespace aspect
{
#define INSTANTIATE(dim) \
  template class StokesMatrixFreeHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)
}
