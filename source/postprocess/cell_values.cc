/*
  Copyright (C) 2016 - 2018 by the authors of the ASPECT code.

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


#include <aspect/postprocess/cell_values.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/geometry_model/sphere.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/global.h>
#include <deal.II/numerics/vector_tools.h>

#include <math.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    CellValues<dim>::CellValues ()
      :
      // the following value is later read from the input file
      output_interval (0),
      // initialize this to a nonsensical value; set it to the actual time
      // the first time around we get to check it
      last_output_time (std::numeric_limits<double>::quiet_NaN())
    {}

    template <int dim>
    std::pair<std::string,std::string>
    CellValues<dim>::execute (TableHandler &)
    {
      // if this is the first time we get here, set the next output time
      // to the current time. this makes sure we always produce data during
      // the first time step
      if (std::isnan(last_output_time))
        last_output_time = this->get_time() - output_interval;

      // see if output is requested at this time
      if (this->get_time() < last_output_time + output_interval)
        return std::pair<std::string,std::string>();

      // evaluate the solution at all of our evaluation points
      const unsigned int n_data_points = 2;

      OneTimeStepData current_data;

      current_data.time_step_no = this->get_timestep_number();
      current_data.time = this->get_time();

      std::vector<std::array<double,dim+n_data_points>> my_data;
      {
        const unsigned int quadrature_degree = 5;
        const QMidpoint<1> quadrature_formula_mp;
        const QIterated<dim> quadrature_formula(quadrature_formula_mp, quadrature_degree);
        const unsigned int n_q_points = quadrature_formula.size(); // this is 1 for QMidpoint

        FEValues<dim> fe_values(this->get_mapping(), this->get_fe(),
                                quadrature_formula,
                                update_values | update_gradients | update_quadrature_points | update_JxW_values);

        typename DoFHandler<dim>::active_cell_iterator
        cell = this->get_dof_handler().begin_active(),
        endc = this->get_dof_handler().end();

        const double middle = 5000.;

        std::vector<SymmetricTensor<2,dim> > strain_rates(n_q_points);
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                if (cell->center()[1] < middle || cell->center()[1]>middle+0.5*cell->diameter())
                  continue; // only output cells in the "middle"

                fe_values.reinit(cell);
                fe_values[this->introspection().extractors.velocities].get_function_symmetric_gradients (this->get_solution(), strain_rates);

                std::array<double, dim+2> data;
                const Point<dim> position = cell->center();
                for (unsigned int d=0; d<dim; ++d)
                  data[d] = position[d];

                double area = 0.0;
                double incompressible_strain_rate_value = 0.0;
                double compressible_strain_rate_value = 0.0;

                for (unsigned int q=0; q<n_q_points; ++q)
                  {
                    const SymmetricTensor<2,dim> strain_rate = strain_rates[q];

                    incompressible_strain_rate_value +=
                      std::sqrt(strain_rate * strain_rate)
                      * fe_values.JxW(q);

                    const SymmetricTensor<2,dim> compressible_strain_rate
                      = (this->get_material_model().is_compressible()
                         ?
                         strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                         :
                         strain_rate);

                    compressible_strain_rate_value += std::sqrt(compressible_strain_rate*compressible_strain_rate)
                                                      * fe_values.JxW(q);

                    area += fe_values.JxW(q);

                  }

                data[dim+0] = incompressible_strain_rate_value;
                data[dim+1] = compressible_strain_rate_value;

                my_data.push_back(data);
              }

          }



        {
          // transfer data to rank 0
          const int my_rank = dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
          const int n_proc = dealii::Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());

          MPI_Barrier(this->get_mpi_communicator());

          if (my_rank == 0)
            {
              for (int proc = 1; proc<n_proc; ++proc)
                {
                  MPI_Status status;
                  MPI_Probe(MPI_ANY_SOURCE, 123, this->get_mpi_communicator(), &status);
                  int count;
                  MPI_Get_count(&status, MPI_DOUBLE, &count);

                  std::vector<std::array<double,dim+n_data_points>> recv_data(count/(dim+n_data_points));
                  MPI_Recv(recv_data.data(), count, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, this->get_mpi_communicator(), MPI_STATUS_IGNORE);
                  my_data.insert(my_data.end(), recv_data.begin(), recv_data.end());
                }
              std::sort(my_data.begin(), my_data.end());
              for (const auto &it : my_data)
                {
                  Vector<double> x(dim+n_data_points);
                  for (unsigned int i=0; i<dim+n_data_points; ++i)
                    x[i] = it[i];
                  current_data.data.push_back(x);
                }
              data.push_back(current_data);
            }
          else
            {
              int count = my_data.size()*(dim+n_data_points);
              MPI_Send(my_data.data(), count, MPI_DOUBLE, 0, 123, this->get_mpi_communicator());
            }
        }
      }

      // now write all of the data to the file of choice. start with a pre-amble that
      // explains the meaning of the various fields
      const std::string filename = (this->get_output_directory() +
                                    "cell_values.txt");
      std::ofstream f (filename.c_str());
      f << ("# <time_step_no> <time> <p x> <p y> ")
        << (dim == 3 ? "<p z> " : "")
        << ("<SR> <SR_dev> ");
      f << '\n';

      for (const auto &it : data)
        {
          for (unsigned int i=0; i<it.data.size(); ++i)
            {
              f << it.time_step_no
                << ' ' << it.time;
              f << ' ' << it.data[i];
            }

          // have an empty line between time steps
          f << '\n';
        }

      AssertThrow (f, ExcMessage("Writing data to <" + filename +
                                 "> did not succeed in the `point values' "
                                 "postprocessor."));

      // Update time
      set_last_output_time (this->get_time());

      // return what should be printed to the screen. note that we had
      // just incremented the number, so use the previous value
      return std::make_pair (std::string ("Writing cell values:"),
                             filename);
    }


    template <int dim>
    void
    CellValues<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Cell values");
        {
          prm.declare_entry ("Time between point values output", "0",
                             Patterns::Double (0),
                             "The time interval between each generation of "
                             "point values output. A value of zero indicates "
                             "that output should be generated in each time step. "
                             "Units: years if the "
                             "'Use years in output instead of seconds' parameter is set; "
                             "seconds otherwise.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    CellValues<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Cell values");
        {
          output_interval = prm.get_double ("Time between point values output");
          if (this->convert_output_to_years())
            output_interval *= year_in_seconds;
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    template <class Archive>
    void CellValues<dim>::serialize (Archive &/*ar*/, const unsigned int)
    {
      /*ar &evaluation_points_cartesian
      & point_values
      & last_output_time;*/
    }


    template <int dim>
    void
    CellValues<dim>::save (std::map<std::string, std::string> &status_strings) const
    {
      std::ostringstream os;
      aspect::oarchive oa (os);
      oa << (*this);

      status_strings["PointValues"] = os.str();
    }


    template <int dim>
    void
    CellValues<dim>::load (const std::map<std::string, std::string> &status_strings)
    {
      // see if something was saved
      if (status_strings.find("PointValues") != status_strings.end())
        {
          std::istringstream is (status_strings.find("PointValues")->second);
          aspect::iarchive ia (is);
          ia >> (*this);
        }
    }


    template <int dim>
    void
    CellValues<dim>::set_last_output_time (const double current_time)
    {
      // if output_interval is positive, then set the next output interval to
      // a positive multiple.
      if (output_interval > 0)
        {
          // We need to find the last time output was supposed to be written.
          // this is the last_output_time plus the largest positive multiple
          // of output_intervals that passed since then. We need to handle the
          // edge case where last_output_time+output_interval==current_time,
          // we did an output and std::floor sadly rounds to zero. This is done
          // by forcing std::floor to round 1.0-eps to 1.0.
          const double magic = 1.0+2.0*std::numeric_limits<double>::epsilon();
          last_output_time = last_output_time + std::floor((current_time-last_output_time)/output_interval*magic) * output_interval/magic;
        }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(CellValues,
                                  "cell values",
                                  "A postprocessor that evaluates the solution (i.e., velocity, pressure, "
                                  "temperature, and compositional fields along with other fields that "
                                  "are treated as primary variables) at the end of every time step or "
                                  "after a user-specified time interval "
                                  "at a given set of points and then writes this data into the file "
                                  "<point\\_values.txt> in the output directory. The points at which "
                                  "the solution should be evaluated are specified in the section "
                                  "\\texttt{Postprocess/Point values} in the input file."
                                  "\n\n"
                                  "In the output file, data is organized as (i) time, (ii) the 2 or 3 "
                                  "coordinates of the evaluation points, and (iii) followed by the "
                                  "values of the solution vector at this point. The time is provided "
                                  "in seconds or, if the "
                                  "global ``Use years in output instead of seconds'' parameter is "
                                  "set, in years. In the latter case, the velocity is also converted "
                                  "to meters/year, instead of meters/second."
                                  "\n\n"
                                  "\\note{Evaluating the solution of a finite element field at "
                                  "arbitrarily chosen points is an expensive process. Using this "
                                  "postprocessor will only be efficient if the number of evaluation "
                                  "points or output times is relatively small. If you need a very large number of "
                                  "evaluation points, you should consider extracting this "
                                  "information from the visualization program you use to display "
                                  "the output of the `visualization' postprocessor.}")
  }
}
