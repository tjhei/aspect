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

#include <deal.II/fe/fe_values.h>


namespace aspect
{
  namespace internal
  {

  }

  template <int dim>
  StokesMatrixFreeHandler<dim>::StokesMatrixFreeHandler (Simulator<dim> &simulator,
                                                         ParameterHandler &prm)
    : sim(simulator)
  {
    parse_parameters(prm);
    //TODO CitationInfo::add("mf");
  }

  template <int dim>
  StokesMatrixFreeHandler<dim>::~StokesMatrixFreeHandler ()
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
  }


}


// explicit instantiation of the functions we implement in this file
namespace aspect
{
#define INSTANTIATE(dim) \
  template class StokesMatrixFreeHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)
}
