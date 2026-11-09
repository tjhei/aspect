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
#include <deal.II/dofs/dof_tools.h>

#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace MeshDeformation
  {
    template <int dim>
    class TestBoundaryTransferError : public ParallelUnstructuredInterface<dim>
    {
      public:
        void update() override
        {
          if (this->remote_point_evaluator)
            return;

          // Sample a smooth non-polynomial field on successively finer fixed
          // external grids. Each sample is the center of its rectangular dual
          // cell, whose area supplies the weight. These points deliberately do
          // not coincide with the ASPECT support points.
          const auto comm = this->get_mpi_communicator();
          const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
          const unsigned int ranks = Utilities::MPI::n_mpi_processes(comm);
          this->get_pcout() << "Nodes per axis | nearest L2 error | weighted L2 error | nearest/weighted" << std::endl;
          const unsigned int resolutions[] = {8, 16, 32, 64};
          for (const unsigned int n : resolutions)
            {
              std::vector<Point<dim>> points;
              std::vector<double> weights;
              std::vector<Tensor<1,dim>> values;
              for (unsigned int i=0; i<n; ++i)
                for (unsigned int j=0; j<(dim == 3 ? n : 1u); ++j)
                  if ((i*(dim == 3 ? n : 1u)+j)%ranks == rank)
                    {
                      Point<dim> point;
                      point[0] = (i+0.5)/n;
                      if constexpr (dim == 3)
                        point[1] = (j+0.5)/n;
                      point[dim-1] = 1.;
                      points.push_back(point);
                      weights.push_back(dim == 3 ? 1./(n*n) : 1./n);
                      Tensor<1,dim> value;
                      value[dim-1] = exact_value(point);
                      values.push_back(value);
                    }

              // Both methods receive exactly the same sample coordinates and
              // values. Only the transfer method and the use of weights differ.
              this->set_evaluation_points(points);
              const auto nearest = this->interpolate_external_vector_field_to_surface_support_points(values);
              const double nearest_error = boundary_error(nearest);
              this->set_evaluation_points_with_weights(points, weights);
              const auto weighted = this->interpolate_external_vector_field_to_surface_support_points(values);
              const double weighted_error = boundary_error(weighted);

              std::ostringstream output;
              output << n << " | " << std::scientific << std::setprecision(6)
                     << nearest_error << " | " << weighted_error << " | "
                     << std::fixed << std::setprecision(2) << nearest_error/weighted_error;
              this->get_pcout() << output.str() << std::endl;
              // Coarse dual-cell quadrature need not improve the field error.
              // At sufficient external resolution the projection should win.
              if (n == 64)
                AssertThrow(weighted_error < 0.5*nearest_error,
                            ExcMessage("The finest external grid must reduce the weighted-transfer L2 error by at least a factor of two."));
            }
        }

        std::vector<Tensor<1,dim>>
        compute_updated_velocities_at_points(const std::vector<std::vector<double>> &solution) const override
        {
          return std::vector<Tensor<1,dim>>(solution.size());
        }

      private:
        static double exact_value(const Point<dim> &point)
        {
          return std::exp(point[0] + (dim == 3 ? 0.5*point[1] : 0.));
        }

        double boundary_error(const LinearAlgebra::Vector &field) const
        {
          const auto &dofs = this->get_mesh_deformation_handler().get_mesh_deformation_dof_handler();
          const auto comm = this->get_mpi_communicator();
          LinearAlgebra::Vector ghosted(dofs.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(dofs), comm);
          ghosted = field;
          const MappingQ<dim> mapping(1);
          // Use an independent, high-order integration rule for the error;
          // this is neither the external sampling rule nor the mass rule.
          FEFaceValues<dim> face_values(mapping, dofs.get_fe(), QGauss<dim-1>(8),
                                        update_values | update_quadrature_points | update_JxW_values);
          std::vector<Tensor<1,dim>> face_field(face_values.n_quadrature_points);
          double squared_error = 0.;
          for (const auto &cell : dofs.active_cell_iterators())
            if (cell->is_locally_owned())
              for (const auto face : cell->face_indices())
                if (cell->face(face)->at_boundary() && this->get_mesh_deformation_boundary_indicators().count(cell->face(face)->boundary_id()))
                  {
                    face_values.reinit(cell, face);
                    face_values[FEValuesExtractors::Vector(0)].get_function_values(ghosted, face_field);
                    for (unsigned int q=0; q<face_field.size(); ++q)
                      {
                        Tensor<1,dim> exact;
                        exact[dim-1] = exact_value(face_values.quadrature_point(q));
                        squared_error += (face_field[q]-exact).norm_square()*face_values.JxW(q);
                      }
                  }
          return std::sqrt(Utilities::MPI::sum(squared_error, comm));
        }
    };
    ASPECT_REGISTER_MESH_DEFORMATION_MODEL(TestBoundaryTransferError, "test boundary transfer error", "")
  }
}
