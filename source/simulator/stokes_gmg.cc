/*
  Copyright (C) 2019 by the authors of the ASPECT code.

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
#include <aspect/global.h>
#include <aspect/melt.h>

#include <deal.II/base/signaling_nan.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace internal
  {

  }

  template <int dim>
  std::pair<double,double>
  Simulator<dim>::solve_stokes_block_gmg ()
  {
    pcout << "\n   Solving Stokes using GMG " << std::endl;

    // extract Stokes parts of solution vector, without any ghost elements
    LinearAlgebra::BlockVector distributed_stokes_solution (introspection.index_sets.stokes_partitioning, mpi_communicator);

    double initial_nonlinear_residual = numbers::signaling_nan<double>();
    double final_linear_residual      = numbers::signaling_nan<double>();

    Assert(!parameters.include_melt_transport, ExcNotImplemented("Sorry, no."));
    const unsigned int block_vel = introspection.block_indices.velocities;
    const unsigned int block_p = introspection.block_indices.pressure;
    Assert(block_vel == 0, ExcNotImplemented());
    Assert(block_p == 1, ExcNotImplemented());


    // TODO: solve. look at solver.cc solve_stokes()
    
    //you can use dof_handler and current_constraints here

    
    // TODO: signal successful solver
    /*
      signals.post_stokes_solver(*this,
                                   preconditioner_cheap.n_iterations_S() + preconditioner_expensive.n_iterations_S(),
                                   preconditioner_cheap.n_iterations_A() + preconditioner_expensive.n_iterations_A(),
                                   solver_control_cheap,
                                   solver_control_expensive);
    */

        // distribute hanging node and
        // other constraints
        current_constraints.distribute (distributed_stokes_solution);

        // now rescale the pressure back to real physical units
        distributed_stokes_solution.block(block_p) *= pressure_scaling;

        // then copy back the solution from the temporary (non-ghosted) vector
        // into the ghosted one with all solution components
        solution.block(block_vel) = distributed_stokes_solution.block(block_vel);
        solution.block(block_p) = distributed_stokes_solution.block(block_p);

        // print the number of iterations to screen
	/*
        pcout << (solver_control_cheap.last_step() != numbers::invalid_unsigned_int ?
                  solver_control_cheap.last_step():
                  0)
              << '+'
              << (solver_control_expensive.last_step() != numbers::invalid_unsigned_int ?
                  solver_control_expensive.last_step():
                  0)
              << " iterations.";
        pcout << std::endl;
	*/
	
    // do some cleanup now that we have the solution
    remove_nullspace(solution, distributed_stokes_solution);

    //if (!assemble_newton_stokes_system)
    //  this->last_pressure_normalization_adjustment = normalize_pressure(solution);

    // convert melt pressures:
    //if (parameters.include_melt_transport)
    //  melt_handler->compute_melt_variables(solution);

    return std::pair<double,double>(initial_nonlinear_residual,
                                    final_linear_residual);
  }

}





// explicit instantiation of the functions we implement in this file
namespace aspect
{
#define INSTANTIATE(dim) \
  template std::pair<double,double> Simulator<dim>::solve_stokes_block_gmg ();

  ASPECT_INSTANTIATE(INSTANTIATE)
}
