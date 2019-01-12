/*
  Copyright (C) 2018 - 2019 by the authors of the ASPECT code.

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


#ifndef _aspect_stokes_matrix_free__h
#define _aspect_stokes_matrix_free__h

#include <aspect/simulator.h>

namespace aspect
{
  using namespace dealii;

  template<int dim>
  class StokesMatrixFreeHandler
  {
    public:
      /**
       * Initialize this class, allowing it to read in
       * relevant parameters as well as giving it a reference to the
       * Simulator that owns it, since it needs to make fairly extensive
       * changes to the internals of the simulator.
       */
      StokesMatrixFreeHandler(Simulator<dim> &, ParameterHandler &prm);

      /**
       * Destructor for the free surface handler.
       */
      ~StokesMatrixFreeHandler();

      /**
       * The main execution step
       */
      std::pair<double,double> solve();

      /**
       * Allocates and sets up the members of the FreeSurfaceHandler. This
       * is called by Simulator<dim>::setup_dofs()
       */
      void setup_dofs();

      /**
       * Declare parameters.
       */
      static
      void declare_parameters (ParameterHandler &prm);

      /**
       * Parse parameters
       */
      void parse_parameters (ParameterHandler &prm);

    private:
      /**
       * Reference to the Simulator object to which a FreeSurfaceHandler
       * instance belongs.
       */
      Simulator<dim> &sim;

      //friend class Simulator<dim>;
      //friend class SimulatorAccess<dim>;
  };
}


#endif
