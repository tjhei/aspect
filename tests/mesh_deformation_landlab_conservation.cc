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

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/mesh_deformation/interface.h>
#include <aspect/geometry_model/interface.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>
#include <iomanip>
#include <sstream>

namespace aspect
{
  template <int dim>
  class LandlabDepositionConservation : public Postprocess::Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        const auto comm = this->get_mpi_communicator();
        std::vector<double> expected(3, std::numeric_limits<double>::quiet_NaN());
        if (Utilities::MPI::this_mpi_process(comm) == 0)
          {
            std::ifstream input(this->get_output_directory() + "landlab_deposition.txt");
            input >> expected[0] >> expected[1] >> expected[2];
            AssertThrow(input, ExcMessage("Could not read the Landlab deposition totals."));
          }
        expected = Utilities::MPI::broadcast(comm, expected, 0);
        AssertThrow(expected[0] == this->get_timestep_number(),
                    ExcMessage("The Landlab deposition totals are from a different timestep."));

        const auto &handler = this->get_mesh_deformation_handler();
        const auto &dofs = handler.get_mesh_deformation_dof_handler();
        LinearAlgebra::Vector displacement(dofs.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(dofs), comm);
        displacement = handler.get_mesh_displacements();
        const MappingQ<dim> reference_mapping(1);
        FEFaceValues<dim> face_values(reference_mapping, dofs.get_fe(), QGauss<dim-1>(dofs.get_fe().degree+1),
                                      update_values | update_JxW_values);
        std::vector<Tensor<1,dim>> face_displacement(face_values.n_quadrature_points);
        const auto top = this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
        double local_volume = 0.;
        for (const auto &cell : dofs.active_cell_iterators())
          if (cell->is_locally_owned())
            for (const auto face : cell->face_indices())
              if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == top)
                {
                  face_values.reinit(cell, face);
                  face_values[FEValuesExtractors::Vector(0)].get_function_values(displacement, face_displacement);
                  for (unsigned int q=0; q<face_displacement.size(); ++q)
                    local_volume += face_displacement[q][dim-1]*face_values.JxW(q);
                }
        const double volume = Utilities::MPI::sum(local_volume, comm);
        const double received = volume - previous_volume;
        const double tolerance = 1e-13 + 1e-9*std::abs(expected[2]);
        std::ostringstream diagnostic;
        diagnostic << std::setprecision(17) << "Landlab step total=" << expected[1]
                   << ", ASPECT step total=" << received
                   << ", difference=" << received-expected[1];
        AssertThrow(std::abs(received-expected[1]) < tolerance,
                    ExcMessage(diagnostic.str()));
        AssertThrow(std::abs(volume-expected[2]) < tolerance,
                    ExcMessage("ASPECT's accumulated boundary displacement does not match Landlab's cumulative deposition."));
        previous_volume = volume;

        std::ostringstream output;
        output << std::scientific << std::setprecision(8)
               << "Landlab=" << expected[1] << ", ASPECT=" << received
               << ", cumulative=" << volume;
        return {"Deposition conservation:", output.str()};
      }

    private:
      double previous_volume = 0.;
  };
  ASPECT_REGISTER_POSTPROCESSOR(LandlabDepositionConservation, "landlab deposition conservation", "")
}
