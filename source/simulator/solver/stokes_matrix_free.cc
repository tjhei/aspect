/*
  Copyright (C) 2011 - 2025 by the authors of the ASPECT code.

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
#include <aspect/simulator/solver/stokes_matrix_free.h>
#include <aspect/simulator/solver/matrix_free_operators.h>
#include <aspect/simulator/solver/stokes_matrix_free_local_smoothing.h>
#include <aspect/simulator/solver/stokes_matrix_free_global_coarsening.h>

namespace aspect
{
  template <int dim>
  void
  StokesMatrixFreeHandler<dim>::setup_multigrid_hierarchy()
  {}

  template <int dim>
  std::unique_ptr<StokesMatrixFreeHandler<dim>> create_matrix_free_solver(Simulator<dim> &simulator, const Parameters<dim> &parameters)
  {
    if (parameters.stokes_gmg_type == Parameters<dim>::StokesGMGType::local_smoothing)
      {
        switch (parameters.stokes_velocity_degree)
          {
            case 2:
              return std::make_unique<StokesMatrixFreeHandlerLocalSmoothingImplementation<dim,2>>(simulator, parameters);
              break;
            case 3:
              return std::make_unique<StokesMatrixFreeHandlerLocalSmoothingImplementation<dim,3>>(simulator, parameters);
              break;
            default:
              AssertThrow(false, ExcMessage("The finite element degree for the Stokes system you selected is not supported yet."));
          }
      }
    else if (parameters.stokes_gmg_type == Parameters<dim>::StokesGMGType::global_coarsening)
      {
        switch (parameters.stokes_velocity_degree)
          {
            case 2:
              return std::make_unique<StokesMatrixFreeHandlerGlobalCoarseningImplementation<dim,2>>(simulator, parameters);
              break;
            case 3:
              return std::make_unique<StokesMatrixFreeHandlerGlobalCoarseningImplementation<dim,3>>(simulator, parameters);
              break;
            default:
              AssertThrow(false, ExcMessage("The finite element degree for the Stokes system you selected is not supported yet."));
          }
      }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  template <int dim>
  void StokesMatrixFreeHandler<dim>::declare_parameters(ParameterHandler &prm)
  {
    StokesMatrixFreeHandlerLocalSmoothingImplementation<dim,2>::declare_parameters(prm);
    StokesMatrixFreeHandlerGlobalCoarseningImplementation<dim,2>::declare_parameters(prm);
  }

  template <int dim>
  void StokesMatrixFreeHandler<dim>::
  compute_pressure_nullspace_flag()
  {
    const auto &bvm = this->get_boundary_velocity_manager();

    // boundaries whose normal velocity is constrained
    std::set<types::boundary_id> constrained = bvm.get_zero_boundary_velocity_indicators();
    const auto &tang = bvm.get_tangential_boundary_velocity_indicators();
    constrained.insert(tang.begin(), tang.end());

    // prescribed boundaries: an empty mask means all components (same convention as setup_dofs)
    std::set<types::boundary_id> masked;
    for (const auto id : bvm.get_prescribed_boundary_velocity_indicators())
      {
        const ComponentMask m = bvm.get_component_mask(id);
        if (m == ComponentMask(this->introspection().n_components, false))
          constrained.insert(id);
        else
          masked.insert(id);
      }

    // periodic faces are interior, not boundaries
    std::set<types::boundary_id> periodic;
    for (const auto &pp : this->get_geometry_model().get_periodic_boundary_pairs())
      {
        periodic.insert(pp.first.first);
        periodic.insert(pp.first.second);
      }

    // masked boundary: normal is constrained iff every component with a nonzero normal entry is masked.
    // Uses fe_normals and the triangulation only, so it works before distribute_dofs().
    std::map<types::boundary_id,int> masked_ok;
    for (const auto id : masked) masked_ok[id] = 1;

    if (!masked.empty())
      {
        const FE_Q<dim> fe_normals(1);
        FEFaceValues<dim> fe_face(this->get_mapping(), fe_normals,
                                  QMidpoint<dim-1>(), update_normal_vectors);

        for (const auto &cell : this->get_triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            for (const auto f : cell->face_indices())
              if (cell->face(f)->at_boundary()
                  && masked.count(cell->face(f)->boundary_id()))
                {
                  const auto id = cell->face(f)->boundary_id();
                  fe_face.reinit(cell, f);
                  const Tensor<1,dim> n = fe_face.normal_vector(0);
                  const ComponentMask m = bvm.get_component_mask(id);
                  for (unsigned int c=0; c<dim; ++c)
                    if (!m[this->introspection().component_indices.velocities[c]]
                        && std::abs(n[c]) > 1e-8)
                      masked_ok[id] = 0;
                }
      }

    // Every rank must take part in the reduction, whether or not it owns
    // any masked face. The map has the same keys on all ranks.
    for (auto &kv : masked_ok)
      kv.second = Utilities::MPI::min(kv.second, this->get_mpi_communicator());

    free_velocity_boundaries.clear();
    for (const auto id : this->get_geometry_model().get_used_boundary_indicators())
      {
        if (periodic.count(id) || constrained.count(id)) continue;
        if (masked_ok.count(id) && masked_ok[id] == 1) continue;
        free_velocity_boundaries.insert(id);
      }
    pressure_laplace_has_nullspace = free_velocity_boundaries.empty();
    // ---- DEBUG diagnostic: how was each boundary classified? ----
    for (const auto id : this->get_geometry_model().get_used_boundary_indicators())
      {
        std::string kind;
        if (periodic.count(id))                                     kind = "periodic (interior)";
        else if (bvm.get_zero_boundary_velocity_indicators().count(id)) kind = "zero velocity (no-slip)";
        else if (tang.count(id))                                    kind = "tangential (free-slip)";
        else if (masked.count(id))                                  kind = (masked_ok[id] == 1)
                                                                             ? "prescribed, masked (normal constrained)"
                                                                             : "prescribed, masked (normal FREE)";
        else if (constrained.count(id))                             kind = "prescribed (all components)";
        else                                                        kind = "FREE (traction/open)";

        this->get_pcout() << "  boundary " << id << " ("
                          << this->get_geometry_model().translate_id_to_symbol_name(id)
                          << "): " << kind << std::endl;
      }
    this->get_pcout() << "  pressure Laplace has nullspace: "
                      << (pressure_laplace_has_nullspace ? "yes (remove mean)"
                          : "no (Dirichlet p=0 on free boundaries)")
                      << std::endl;
  }
}

// explicit instantiations
namespace aspect
{
#define INSTANTIATE(dim) \
  template std::unique_ptr<StokesMatrixFreeHandler<dim>> create_matrix_free_solver(Simulator<dim> &, const Parameters<dim> &); \
  template class StokesMatrixFreeHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
