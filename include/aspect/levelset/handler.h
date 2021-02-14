/*
  Copyright (C) 2020 - 2021 by the authors of the ASPECT code.

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

#ifndef _aspect_levelset_handler_h
#define _aspect_levelset_handler_h

#include <aspect/simulator_access.h>
#include <aspect/simulator.h>

using namespace dealii;

namespace aspect
{
  template <int dim> class Simulator;

  /**
   * A member class that isolates the functions and variables that deal
   * with the Volume of Fluid implementation. If Volume of Fluid interface
   * tracking is not active, there is no instantiation of this class at
   * all.
   */
  template <int dim>
  class LevelsetHandler : public SimulatorAccess<dim>
  {
    public:
      /**
       * Standard initial constructor
       */
      LevelsetHandler(Simulator<dim> &simulator, ParameterHandler &prm);

      /**
       * Declare the parameters this class takes through input files.
       */
      static
      void declare_parameters (ParameterHandler &prm);

      /**
       * Read the parameters this class declares from the parameter file.
       */
      void parse_parameters (ParameterHandler &prm);

      /**
       * Do necessary internal initialization that is dependent on having the
       * simulator and Finite Element initialized.
       */
      void initialize (ParameterHandler &prm);

      /**
       * Perform the advection step for the specified field.
       */
      void solve(const typename Simulator<dim>::AdvectionField &advection_field);

    private:
      /**
       * Parent simulator
       */
      Simulator<dim> &sim;


      friend class Simulator<dim>;
  };

}

#endif
