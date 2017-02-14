/*
  Copyright (C) 2016 by the authors of the ASPECT code.

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


#include <aspect/postprocess/temperature_ascii_out.h>
#include <aspect/global.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>

#include <math.h>
#include <algorithm>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    TemperatureAsciiOut<dim>::TemperatureAsciiOut()
    {}

    template <int dim>
    std::pair<std::string,std::string>
    TemperatureAsciiOut<dim>::execute (TableHandler &)
    {
      std::string filename = (this->get_output_directory() + "temperature_ascii_data.txt");

      struct entry
      {
        Point<dim> p;
        double t;
      };
      std::vector<entry> entries;

      //      const QTrapez<dim> quadrature_formula;
      const QMidpoint<dim> quadrature_formula;

      const unsigned int n_q_points =  quadrature_formula.size();
      FEValues<dim> fe_values (this->get_mapping(), this->get_fe(),  quadrature_formula,
                               update_JxW_values | update_values | update_quadrature_points);

      std::vector<double> temperature_values(n_q_points);

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();
      for (; cell != endc; ++cell)
        {
          fe_values.reinit (cell);
          fe_values[this->introspection().extractors.temperature].get_function_values (this->get_solution(), temperature_values);

          for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
            {
              entry e;
              e.p = fe_values.quadrature_point(q);
              e.t = temperature_values[q];
              entries.push_back(e);
            }
        }

      struct sorter
      {
        bool operator() (const entry &i, const entry &j)
        {
          if (std::abs(i.p[1]-j.p[1])<1e-6)
            return i.p[0] < j.p[0];
          else
            return i.p[1] < j.p[1];
        }
      } sorter_instance;
      /*      struct comparer
      {
        bool operator() (const entry &i, const entry &j)
        {
          return (std::abs(i.p[0]-j.p[0])<1e-6)
      && (std::abs(i.p[1]-j.p[1])<1e-6);
        }
      } comparer_instance;*/
      std::sort(entries.begin(), entries.end(), sorter_instance);
      //      entries.erase( std::unique( entries.begin(), entries.end(), comparer_instance ), entries.end() );



      unsigned int n_x, n_y;
      {
        // find unique number of x and y values by stuffing them into a std::set
        struct sort_double
        {
          bool operator() (const double &i, const double &j)
          {
            return i<j;
          }
        } sort_double_instance;
        struct compare_double_eps
        {
          bool operator() (const double &i, const double &j)
          {
            return (std::abs(i-j)<1e-6);
          }
        } compare_double_instance;

        std::vector<double> x_values, y_values;
        for (unsigned int idx = 0; idx< entries.size(); ++idx)
          {
            x_values.push_back(entries[idx].p(0));
            y_values.push_back(entries[idx].p(1));
          }
        std::sort(x_values.begin(), x_values.end(), sort_double_instance);
        x_values.erase( std::unique( x_values.begin(), x_values.end(), compare_double_instance ), x_values.end() );
        std::sort(y_values.begin(), y_values.end(), sort_double_instance);
        y_values.erase( std::unique( y_values.begin(), y_values.end(), compare_double_instance ), y_values.end() );

        n_x = x_values.size();
        n_y = y_values.size();
      }


      std::ofstream f(filename.c_str());

      f << std::scientific << std::setprecision(15);

      // Note: POINTS is only useful if our mesh is a structured grid
      f << "# POINTS: " << n_x << " " << n_y << "\n";
      f << "# x y T\n";

      for (unsigned int idx = 0; idx< entries.size(); ++idx)
        f << entries[idx].p << ' ' << entries[idx].t << '\n';

      f.close();

      // return what should be printed to the screen. note that we had
      // just incremented the number, so use the previous value
      return std::make_pair (std::string ("Writing TemperatureAsciiOut:"),
                             filename);
    }


    template <int dim>
    void
    TemperatureAsciiOut<dim>::declare_parameters (ParameterHandler &prm)
    {
    }


    template <int dim>
    void
    TemperatureAsciiOut<dim>::parse_parameters (ParameterHandler &prm)
    {
    }


    template <int dim>
    template <class Archive>
    void TemperatureAsciiOut<dim>::serialize (Archive &ar, const unsigned int)
    {
      //ar &evaluation_points
      //& point_values;
    }


    template <int dim>
    void
    TemperatureAsciiOut<dim>::save (std::map<std::string, std::string> &status_strings) const
    {
      std::ostringstream os;
      aspect::oarchive oa (os);
      oa << (*this);

      status_strings["PointValues"] = os.str();
    }


    template <int dim>
    void
    TemperatureAsciiOut<dim>::load (const std::map<std::string, std::string> &status_strings)
    {
      // see if something was saved
      if (status_strings.find("PointValues") != status_strings.end())
        {
          std::istringstream is (status_strings.find("PointValues")->second);
          aspect::iarchive ia (is);
          ia >> (*this);
        }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(TemperatureAsciiOut,
                                  "temperature ascii out",
                                  "A postprocessor that evaluates ")
  }
}
