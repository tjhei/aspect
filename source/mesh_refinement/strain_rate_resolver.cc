/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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


#include <aspect/mesh_refinement/strain_rate_resolver.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/derivative_approximation.h>

namespace aspect
{
  namespace MeshRefinement
  {
    template <int dim>
    void
    StrainRateResolver<dim>::tag_additional_cells() const
    {
      this->should_repeat = false;
      if (this->get_dof_handler().n_dofs()==0)
        return; // initial global refinement

      const double threshold = 1e-15;
      const int target_level = 5;

      const QMidpoint<dim> quadrature;

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature,
                               update_quadrature_points | update_values | update_gradients);

      std::vector<SymmetricTensor<2,dim> > strain_rates (quadrature.size());

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();
      unsigned int j=0;
      for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            fe_values[this->introspection().extractors.velocities].get_function_symmetric_gradients (this->get_solution(),
                strain_rates);

            double value = strain_rates[0].norm();

            if (value > threshold && cell->level() < target_level)
              {
                cell->clear_coarsen_flag();
                cell->set_refine_flag();
                this->should_repeat = true;
                ++j;
              }
          }

      std::cout << "flag cells" << j << std::endl;
    }

    template <int dim>
    bool
    StrainRateResolver<dim>::should_repeat_time_step() const
    {
      return this->should_repeat;
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MeshRefinement
  {
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(StrainRateResolver,
                                              "strain rate resolver",
                                              "A mesh refinement criterion that ")
  }
}
