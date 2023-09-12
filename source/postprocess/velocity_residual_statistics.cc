/*
  Copyright (C) 2011 - 2023 by the authors of the ASPECT code.

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



#include <aspect/postprocess/velocity_residual_statistics.h>
#include <aspect/utilities.h>
#include <aspect/geometry_model/interface.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    VelocityResidualStatistics<dim>::VelocityResidualStatistics ()
      :
      // the following value is later read from the input file
      output_interval (0),
      // initialize this to a nonsensical value; set it to the actual time
      // the first time around we get to check it
      last_output_time (std::numeric_limits<double>::quiet_NaN()),
      point_values (std::vector<std::pair<double, std::vector<Vector<double>>>>() ),
      output_file_number (numbers::invalid_unsigned_int),
      use_natural_coordinates (false)
    {}

    template <int dim>
    void
    VelocityResidualStatistics<dim>::initialize ()
    {
      // The input ascii table contains one column that just represents points id.
      // The scale factor is set to 1 because the columns contain the observed point
      // coordinates and velocities, which we do not want to scale.
      data_lookup = std::make_unique<Utilities::StructuredDataLookup<1>>(1.0);
      data_lookup->load_file(data_directory + data_file_name, this->get_mpi_communicator());
    }



    template <int dim>
    std::pair <Point<dim>, Tensor<1,dim>>
    VelocityResidualStatistics<dim>::get_observed_data (const unsigned int p) const
    {
      Tensor<1,dim> data_velocity;
      Point<dim> data_position;
      Point<1> point;
      point[0] = p;

      for (unsigned int d = 0; d < dim; ++d)
      {
        data_position[d] = data_lookup->get_data(point, d);
        data_velocity[d] = data_lookup->get_data(point, d+3);
      }

      // if (this->get_geometry_model().natural_coordinate_system() == Utilities::Coordinates::spherical)
      //   {
      //     const std::array<double,dim> cartesian_position = this->get_geometry_model().
      //                                                       natural_to_cartesian_coordinates(data_position);
      //   }
            
      // Since the solution is evaluated in cartesian system, convert the spherical velocities into
      // cartesian
      // if (use_spherical_unit_vectors == true)
      //   data_velocity = Utilities::Coordinates::spherical_to_cartesian_vector(data_velocity, cartesian_position);

      return std::make_pair (data_position, data_velocity);
    }



    template <int dim>
    std::pair<std::string,std::string>
    VelocityResidualStatistics<dim>::execute (TableHandler &statistics)
    {
      // if this is the first time we get here, set the next output time
      // to the current time. this makes sure we always produce data during
      // the first time step
      if (std::isnan(last_output_time))
        last_output_time = this->get_time() - output_interval;

      // see if output is requested at this time
      if (this->get_time() < last_output_time + output_interval)
        return {"", ""};

       // TODO: Modify here for inversion.
      const bool increase_file_number = (this->get_nonlinear_iteration() == 0) ||
                                        (!this->get_parameters().run_postprocessors_on_nonlinear_iterations);
      if (output_file_number == numbers::invalid_unsigned_int)
        output_file_number = 0;
      else if (increase_file_number)
        ++output_file_number;

      const std::string file_prefix = "velocity-residual-" + Utilities::int_to_string (output_file_number, 5);
      const std::string filename = (this->get_output_directory()
                                    + "output_velocity_residual/"
                                    + file_prefix);
      
      unsigned int n_cols;
      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
        {
          // We do not need to distribute the contents as we are using shared data
          // to place it later. Therefore, just pass MPI_COMM_SELF (i.e.,
          // a communicator with just a single MPI process) and no distribution
          // will happen.
          std::ifstream in(data_directory + data_file_name);

          // Read header lines and table size
          while (in.peek() == '#')
            {
              std::string line;
              std::getline(in,line);
              std::stringstream linestream(line);
              std::string word;
              while (linestream >> word)
                if (word == "POINTS:")
                  linestream >> n_cols;
            }
        }

      // evaluate the solution at all of our evaluation points
      std::vector<Vector<double>>
      current_point_values (n_cols, Vector<double> (this->introspection().n_components));

      for (unsigned int p=0; p<n_cols; ++p)
        {
          // try to evaluate the solution at this point. in parallel, the point
          // will be on only one processor's owned cells, so the others are
          // going to throw an exception. make sure at least one processor
          // finds the given point
          bool point_found = false;

          try
            {
              VectorTools::point_value(this->get_mapping(),
                                       this->get_dof_handler(),
                                       this->get_solution(),
                                       get_observed_data(p).first,
                                       current_point_values[p]);
              point_found = true;
            }
          catch (const VectorTools::ExcPointNotAvailableHere &)
            {
              // ignore
            }

          // ensure that at least one processor found things
          const int n_procs = Utilities::MPI::sum (point_found ? 1 : 0, this->get_mpi_communicator());
          AssertThrow (n_procs > 0,
                       ExcMessage ("While trying to evaluate the solution at point " +
                                   Utilities::to_string(get_observed_data(p).first[0]) + ", " +
                                   Utilities::to_string(get_observed_data(p).first[1]) +
                                   (dim == 3
                                    ?
                                    ", " + Utilities::to_string(get_observed_data(p).first[2])
                                    :
                                    "") + "), " +
                                   "no processors reported that the point lies inside the " +
                                   "set of cells they own. Are you trying to evaluate the " +
                                   "solution at a point that lies outside of the domain?"
                                  ));

          // Reduce all collected values into local Vector
          Utilities::MPI::sum (current_point_values[p], this->get_mpi_communicator(),
                               current_point_values[p]);

          // Normalize in cases where points are claimed by multiple processors
          if (n_procs > 1)
            current_point_values[p] /= n_procs;
        }

      // finally push these point values all onto the list we keep
      point_values.emplace_back (this->get_time(), current_point_values);
      
      for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
      {
        this->introspection().name_for_compositional_index(c);
      }

      // create a quadrature formula for the velocity.
      const Quadrature<dim> &quadrature_formula = this->introspection().quadratures.velocities;
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_quadrature_points |
                               update_JxW_values);

      std::vector<Tensor<1,dim>> velocity_values(n_q_points);

      // compute the maximum, minimum, and squared*area velocity residual
      // magnitude and the face area.
      double local_vel_residual_square_integral = 0;
      double local_max_vel_residual             = std::numeric_limits<double>::lowest();
      double local_min_vel_residual             = std::numeric_limits<double>::max();
      double local_fe_volume                    = 0.0;

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);

            fe_values[this->introspection().extractors.velocities].get_function_values (this->get_solution(),
                                                                                        velocity_values);
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const Point<dim> data_point = get_observed_data(q).first;
                Tensor<1,dim> data_velocity = get_observed_data(q).second;

                const double vel_residual_mag = (velocity_values[q] - data_velocity).norm();

                local_max_vel_residual = std::max(vel_residual_mag, local_max_vel_residual);
                local_min_vel_residual = std::min(vel_residual_mag, local_min_vel_residual);

                local_vel_residual_square_integral += ((vel_residual_mag * vel_residual_mag) * fe_values.JxW(q));
                local_fe_volume += fe_values.JxW(q);
              }
          }

      const double global_vel_res_square_integral
        = Utilities::MPI::sum (local_vel_residual_square_integral, this->get_mpi_communicator());
      const double global_max_vel_residual
        = Utilities::MPI::max (local_max_vel_residual, this->get_mpi_communicator());
      const double global_min_vel_residual
        = Utilities::MPI::min (local_min_vel_residual, this->get_mpi_communicator());

      const double vrms_residual = std::sqrt(global_vel_res_square_integral) /
                                   std::sqrt(this->get_volume());

      // now add the computed max, min, and rms velocities to the statistics object
      // and create a single string that can be output to the screen
      const std::string units = (this->convert_output_to_years() == true) ? "m/year" : "m/s";
      const double unit_scale_factor = (this->convert_output_to_years() == true) ? year_in_seconds : 1.0;

      const std::vector<std::string> column_names = {"RMS velocity residual  (" + units + ")",
                                                     "Max. velocity residual (" + units + ")",
                                                     "Min. velocity residual (" + units + ")"
                                                    };

      statistics.add_value (column_names[0],
                            vrms_residual * unit_scale_factor);
      statistics.add_value (column_names[1],
                            global_max_vel_residual * unit_scale_factor);
      statistics.add_value (column_names[2],
                            global_min_vel_residual * unit_scale_factor);

      // also make sure that the other columns filled by this object
      // all show up with sufficient accuracy and in scientific notation
      for (auto &column : column_names)
        {
          statistics.set_precision (column, 8);
          statistics.set_scientific (column, true);
        }

      std::ostringstream screen_text;
      screen_text.precision(3);
      screen_text << vrms_residual *unit_scale_factor
                  << ' ' << units << ", "
                  << global_max_vel_residual *unit_scale_factor
                  << ' ' << units << ", "
                  << global_min_vel_residual *unit_scale_factor
                  << ' ' << units;

      return std::pair<std::string, std::string> ("RMS, max, and min velocity residual velocity in the model:",
                                                  screen_text.str());
    }



    template <int dim>
    void
    VelocityResidualStatistics<dim>::declare_parameters (ParameterHandler  &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Velocity residual statistics");
        {
          prm.declare_entry ("Data directory",
                             "$ASPECT_SOURCE_DIR/data/postprocess/velocity-residual/",
                             Patterns::DirectoryName (),
                             "The name of a directory that contains the input ascii data "
                             "against which the velocity residual in the model is computed. "
                             "This path may either be absolute (if starting with a "
                             "`/') or relative to the current directory. The path may also "
                             "include the special text `$ASPECT_SOURCE_DIR' which will be "
                             "interpreted as the path in which the ASPECT source files were "
                             "located when ASPECT was compiled. This interpretation allows, "
                             "for example, to reference files located in the `data/' subdirectory "
                             "of ASPECT.");
          prm.declare_entry ("Data file name", "box_2d_velocity.txt",
                             Patterns::Anything (),
                             "The file name of the input ascii velocity data. "
                             "The file is provided in the same format as described in "
                             " 'ascii data' initial composition plugin." );
          prm.declare_entry ("Scale factor", "1.",
                             Patterns::Double (),
                             "Scalar factor, which is applied to the model data. "
                             "You might want to use this to scale the input to a "
                             "reference model. Another way to use this factor is to "
                             "convert units of the input files. For instance, if you "
                             "provide velocities in cm/year set this factor to 0.01.");
          prm.declare_entry ("Use spherical unit vectors", "false",
                             Patterns::Bool (),
                             "Specify velocity as r, phi, and theta components "
                             "instead of x, y, and z. Positive velocities point up, east, "
                             "and north (in 3d) or out and clockwise (in 2d). "
                             "This setting only makes sense for spherical geometries.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    VelocityResidualStatistics<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Velocity residual statistics");
        {
          // Get the path to the data files. If it contains a reference
          // to $ASPECT_SOURCE_DIR, replace it by what CMake has given us
          // as a #define
          data_directory = Utilities::expand_ASPECT_SOURCE_DIR(prm.get ("Data directory"));
          data_file_name    = prm.get ("Data file name");
          scale_factor      = 1.0; // prm.get_double ("Scale factor");

          use_spherical_unit_vectors = prm.get_bool("Use spherical unit vectors");

          if (use_spherical_unit_vectors)
            AssertThrow (this->get_geometry_model().natural_coordinate_system() == Utilities::Coordinates::spherical,
                         ExcMessage ("Spherical unit vectors should not be used "
                                     "when geometry model is not spherical."));
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(VelocityResidualStatistics,
                                  "velocity residual statistics",
                                  "A postprocessor that computes some statistics about "
                                  "the velocity residual in the model. The velocity residual "
                                  "is the difference between the model solution velocities and the input "
                                  "ascii data velocities.")
  }
}
