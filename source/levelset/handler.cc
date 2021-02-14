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

#include <aspect/global.h>
#include <aspect/levelset/handler.h>

using namespace dealii;

namespace aspect
{
  namespace internal
  {

  }



  template <int dim>
  LevelsetHandler<dim>::LevelsetHandler (Simulator<dim> &simulator,
                                         ParameterHandler &prm)
    : sim (simulator)
  {
    this->initialize_simulator(sim);
    parse_parameters (prm);
  }



  template <int dim>
  void
  LevelsetHandler<dim>::declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("Levelset");

    prm.leave_subsection();
  }



  template <int dim>
  void
  LevelsetHandler<dim>::parse_parameters (ParameterHandler &prm)
  {

    prm.enter_subsection ("Levelset");
    {

    }
    prm.leave_subsection ();
  }



  template <int dim>
  void
  LevelsetHandler<dim>::initialize (ParameterHandler &/*prm*/)
  {
    AssertThrow(!this->get_material_model().is_compressible(),
                ExcMessage("The levelset method currently assumes incompressibility."));

    AssertThrow(!this->get_parameters().mesh_deformation_enabled,
                ExcMessage("The levelset method is currently incompatible with the Free Surface implementation."));

    AssertThrow(!this->get_parameters().include_melt_transport,
                ExcMessage("The levelset method has not been tested with melt transport yet, so inclusion of both is currently disabled."))

  }


  template <int dim>
  void
  LevelsetHandler<dim>::solve(const typename Simulator<dim>::AdvectionField &advection_field)
  {
    this->get_pcout() << "solving levelset equation..." << std::endl;

    const unsigned int blockidx = advection_field.block_index(sim.introspection);
    LinearAlgebra::Vector field = sim.solution.block(blockidx);

    this->get_pcout() << "    note: "
                      << "old t=" << this->get_time()
                      << " dt=" << this->get_timestep()
                      << " composition component index=" << advection_field.component_index(sim.introspection)
                      << std::endl;

    // TODO: update field

    sim.solution.block(blockidx) = field;

    // In the first timestep we might want to initialize all vectors with the
    // reinitialized levelset
    if (sim.timestep_number == 0)
      {
        // TODO
        //sim.old_solution.block(blockidx) = ...
        // sim.old_old_solution.block(blockidx) = ...
      }

  }

}



namespace aspect
{
#define INSTANTIATE(dim) \
  template class LevelsetHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
