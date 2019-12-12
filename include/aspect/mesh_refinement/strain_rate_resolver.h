/*
  Copyright (C) 2013 - 2017 by the authors of the ASPECT code.

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


#ifndef _aspect_mesh_refinement_strain_rate_h
#define _aspect_mesh_refinement_strain_rate_h

#include <aspect/mesh_refinement/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MeshRefinement
  {

    /**
     * A class that implements a mesh refinement criterion based on
     * the strain rate field.
     *
     * @ingroup MeshRefinement
     */
    template <int dim>
    class StrainRateResolver : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:

        virtual
        void
        tag_additional_cells () const;

        virtual
        bool
        should_repeat_time_step () const;

      private:
        mutable bool should_repeat;
    };
  }
}

#endif
