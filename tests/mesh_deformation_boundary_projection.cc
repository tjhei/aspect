// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the ASPECT authors.
//
// This file is part of ASPECT.
//
// Detailed license information governing the source code and
// contributions can be found in the folder LICENSES and in
// LICENSE and CONTRIBUTING.md at the top level directory.
//
// ------------------------------------------------------

#include <aspect/mesh_deformation/parallel_unstructured_interface.h>
#include <aspect/geometry_model/interface.h>
#include <deal.II/dofs/dof_tools.h>

namespace aspect
{
  namespace MeshDeformation
  {
    template <int dim>
    class TestBoundaryProjection : public ParallelUnstructuredInterface<dim>
    {
      public:
        void update() override
        {
          if (this->remote_point_evaluator)
            return;
          std::vector<Point<dim>> points;
          std::vector<double> weights;
          // Composite Simpson weights integrate the Q2 trace times an affine
          // field on each top face. Middle nodes lie on cell interfaces.
          const double w[] = {1./12, 4./12, 2./12, 4./12, 1./12};
          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
            for (unsigned int i=0; i<5; ++i)
              for (unsigned int j=0; j<(dim == 3 ? 5u : 1u); ++j)
                {
                  Point<dim> p;
                  p[0] = i/4.;
                  if constexpr (dim == 3)
                    p[1] = j/4.;
                  p[dim-1] = 1.;
                  points.push_back(p);
                  weights.push_back(w[i] * (dim == 3 ? w[j] : 1.));
                }
          this->set_evaluation_points_with_weights(points, weights);
          const auto &dofs = this->get_mesh_deformation_handler().get_mesh_deformation_dof_handler();
          const MappingQ<dim> mapping(1);
          const auto support = DoFTools::map_dofs_to_support_points(mapping, dofs);
          const auto boundary = DoFTools::extract_boundary_dofs(dofs, ComponentMask(dim, true),
                                                                this->get_mesh_deformation_boundary_indicators());
          for (unsigned int test=0; test<3; ++test)
            {
              std::vector<Tensor<1,dim>> values(points.size());
              double expected_integral = 0.;
              for (unsigned int q=0; q<points.size(); ++q)
                {
                  const double f = test == 0 ? 1. : test == 1 ? 1.+points[q][0] : std::exp(points[q][0]);
                  values[q][dim-1] = f;
                  expected_integral += weights[q]*f;
                }
              expected_integral = Utilities::MPI::sum(expected_integral, this->get_mpi_communicator());
              auto projected = this->interpolate_external_vector_field_to_surface_support_points(values);
              double error = 0.;
              if (test < 2 && this->get_triangulation().n_global_levels() == 2)
                for (const auto &cell : dofs.active_cell_iterators())
                  if (cell->is_locally_owned())
                    {
                      std::vector<types::global_dof_index> indices(dofs.get_fe().n_dofs_per_cell());
                      cell->get_dof_indices(indices);
                      for (unsigned int i=0; i<indices.size(); ++i)
                        if (dofs.locally_owned_dofs().is_element(indices[i]) && boundary.is_element(indices[i]))
                          {
                            const auto component = dofs.get_fe().system_to_component_index(i).first;
                            const double expected = component == dim-1 ? (test == 0 ? 1. : 1.+support.at(indices[i])[0]) : 0.;
                            error = std::max(error, std::abs(projected[indices[i]]-expected));
                          }
                    }
              LinearAlgebra::Vector ghosted(dofs.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(dofs),
                                            this->get_mpi_communicator());
              ghosted = projected;
              FEFaceValues<dim> face_values(mapping, dofs.get_fe(), QGauss<dim-1>(2), update_values | update_JxW_values);
              std::vector<Tensor<1,dim>> face_field(face_values.n_quadrature_points);
              double integral = 0.;
              for (const auto &cell : dofs.active_cell_iterators())
                if (cell->is_locally_owned())
                  for (const auto face : cell->face_indices())
                    if (cell->face(face)->at_boundary() && this->get_mesh_deformation_boundary_indicators().count(cell->face(face)->boundary_id()))
                      {
                        face_values.reinit(cell, face);
                        face_values[FEValuesExtractors::Vector(0)].get_function_values(ghosted, face_field);
                        for (unsigned int q=0; q<face_field.size(); ++q)
                          integral += face_field[q][dim-1]*face_values.JxW(q);
                      }
              integral = Utilities::MPI::sum(integral, this->get_mpi_communicator());
              error = Utilities::MPI::max(error, this->get_mpi_communicator());
              AssertThrow(error < 1e-10 && std::abs(integral-expected_integral) < 1e-10,
                          ExcMessage("Boundary projection failed accuracy or conservation check."));
              this->get_pcout() << "Boundary projection test " << test << " passed." << std::endl;
            }
        }

        Tensor<1,dim>
        compute_initial_deformation_on_boundary(const types::boundary_id,
                                                const Point<dim> &point) const override
        {
          // Keep a sloping physical surface, so the tests distinguish the
          // fixed reference measure from the deformed surface measure.
          Tensor<1,dim> displacement;
          displacement[dim-1] = 0.01*point[0];
          return displacement;
        }

        std::vector<Tensor<1,dim>> compute_updated_velocities_at_points(const std::vector<std::vector<double>> &solution) const override
        {
          return std::vector<Tensor<1,dim>>(solution.size());
        }
    };
    ASPECT_REGISTER_MESH_DEFORMATION_MODEL(TestBoundaryProjection, "test boundary projection", "")
  }
}
