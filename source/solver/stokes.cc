/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/solver/stokes.h>

namespace aspect
{
  namespace Solver
  {
    template <int dim>
    void
    Extrapolate<dim>::execute (SolveInfo<dim> &info) const
    {
      if (info.timestep_number > 1)
        {
          //TODO: Trilinos sadd does not like ghost vectors even as input. Copy
          //into distributed vectors for now:
          LinearAlgebra::BlockVector & distr_solution = *info.temp_distributed;
          distr_solution = *info.old_solution;
          LinearAlgebra::BlockVector & distr_old_solution = *info.temp_distributed2;
          distr_old_solution = *info.old_old_solution;
          distr_solution.sadd ((1 + info.time_step/info.old_time_step),
                                -info.time_step/info.old_time_step,
                                distr_old_solution);
          *info.current_linearization_point = distr_solution;
        }

    }


    template <int dim>
    class Stokes: public Interface<dim>, public SimulatorAccess<dim>
    {
      public:

        virtual
        void
        execute (SolveInfo<dim> &info) const;
    };

    template <int dim>
    void
    Stokes<dim>::execute (SolveInfo<dim> &info) const
    {

      // the Stokes matrix depends on the viscosity. if the viscosity
      // depends on other solution variables, then after we need to
      // update the Stokes matrix in every time step and so need to set
      // the following flag. if we change the Stokes matrix we also
      // need to update the Stokes preconditioner.
      if (info.simulator->stokes_matrix_depends_on_solution() == true)
        info.simulator->rebuild_stokes_matrix = info.simulator->rebuild_stokes_preconditioner = true;

      info.simulator->assemble_stokes_system();
      info.simulator->build_stokes_preconditioner();
      info.simulator->solve_stokes();
    }


    template <int dim>
    class Temperature: public Interface<dim>, public SimulatorAccess<dim>
    {
      public:

        virtual
        void
        execute (SolveInfo<dim> &info) const;
    };

    template <int dim>
    void
    Temperature<dim>::execute (SolveInfo<dim> &info) const
    {
      assemble_advection_system (AdvectionField::temperature());
      solve_advection(AdvectionField::temperature());

      if (parameters.use_discontinuous_temperature_discretization
          && parameters.use_limiter_for_discontinuous_temperature_solution)
        apply_limiter_to_dg_solutions(AdvectionField::temperature());

      current_linearization_point.block(introspection.block_indices.temperature)
        = solution.block(introspection.block_indices.temperature);
    }


    template <int dim>
    class Compositions: public Interface<dim>, public SimulatorAccess<dim>
    {
      public:

        virtual
        void
        execute (SolveInfo<dim> &info) const;
    };

    template <int dim>
    void
    Compositions<dim>::execute (SolveInfo<dim> &info) const
    {
      for (unsigned int c=0; c < parameters.n_compositional_fields; ++c)
        {
          assemble_advection_system (AdvectionField::composition(c));
          solve_advection(AdvectionField::composition(c));
          if (parameters.use_discontinuous_composition_discretization
              && parameters.use_limiter_for_discontinuous_composition_solution)
            apply_limiter_to_dg_solutions(AdvectionField::composition(c));
        }

      for (unsigned int c=0; c<parameters.n_compositional_fields; ++c)
        current_linearization_point.block(introspection.block_indices.compositional_fields[c])
          = solution.block(introspection.block_indices.compositional_fields[c]);
    }


  } // namespace Solver
} // namespace aspect


// explicit instantiations
namespace aspect
{
  namespace Solver
  {
    ASPECT_REGISTER_SOLVER(Extrapolate,
                                              "extrapolate",
                                              "...")
    ASPECT_REGISTER_SOLVER(Stokes,
                                              "stokes",
                                              "...")
    ASPECT_REGISTER_SOLVER(Temperature,
                                              "temperature",
                                              "...")
    ASPECT_REGISTER_SOLVER(Compositions,
                                              "compositions",
                                              "...")
  }
}
