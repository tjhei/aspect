/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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



#include <aspect/mesh_refinement/interface.h>
#include <aspect/compositional_initial_conditions/interface.h>
#include <aspect/simulator_access.h>



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <math.h>


using namespace dealii;

class Data
    {
      public:


	Data()
	  {
	    sizes[0]=512;
	    sizes[1]=512;
	    sizes[2]=512;
	    data.resize(512*512*512);

	    std::ifstream in("mt_gambier_512.data",std::ios::in | std::ios::binary);
	    in.read((char*)&data[0],512*512*512);
	  }
	

	        void compute_min_max(Point<2> &corner1, Point<2> &corner2, double &minval, double &maxval, double &mean, double &err) const
	    {

	    }
	

        void compute_min_max(Point<3> &corner1, Point<3> &corner2, double &minval, double &maxval, double &mean, double &err) const
    {
      unsigned int x1 = std::min((unsigned int)(sizes[0]*corner1[0]), sizes[0]-1);
      unsigned int x2 = std::min((unsigned int)(sizes[0]*corner2[0]), sizes[0]-1);
      unsigned int y1 = std::min((unsigned int)(sizes[1]*corner1[1]), sizes[1]-1);
      unsigned int y2 = std::min((unsigned int)(sizes[1]*corner2[1]), sizes[1]-1);
      unsigned int z1 = std::min((unsigned int)(sizes[2]*corner1[2]), sizes[2]-1);
      unsigned int z2 = std::min((unsigned int)(sizes[2]*corner2[2]), sizes[2]-1);

      mean = 0.0;
      unsigned int count = 0;
      for (unsigned int x=x1;x<=x2;++x)
        for (unsigned int y=y1;y<=y2;++y)
          for (unsigned int z=z1;z<=z2;++z)
            {
              mean += (double)(data[x+sizes[0]*y + sizes[0]*sizes[1]*z])/255.0;
              count += 1;
            }

      mean /= count;


      err = 0;

      minval = 1.0;
      maxval = 0.0;
      for (unsigned int x=x1;x<=x2;++x)
        for (unsigned int y=y1;y<=y2;++y)
          for (unsigned int z=z1;z<=z2;++z)
            {
              double val = (double)(data[x+sizes[0]*y + sizes[0]*sizes[1]*z])/255.0;
              minval = std::min(minval, val);
              maxval = std::max(maxval, val);
              err += (val-mean)*(val-mean);
            }

      //err /= count;
      //std::cout << minval << " " << maxval << " " << mean << " " << err << std::endl;

      mean = err;

    }

        unsigned int sizes[3];
        std::vector<unsigned char> data;
	
	
    };

Data data;


    
namespace aspect
{
  namespace MeshRefinement
  {


	    

    
    template <int dim>
    class TomoAMR : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        /**
         * After cells have been marked for coarsening/refinement, apply
         * additional criteria independent of the error estimate.
         *
         */
        virtual
        void
        tag_additional_cells () const;

        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter
         * file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:
    };

    
    template <int dim>
    void
    TomoAMR<dim>::tag_additional_cells () const
    {
      for (typename Triangulation<dim>::active_cell_iterator
           cell = this->get_triangulation().begin_active();
           cell != this->get_triangulation().end(); ++cell)
        {
          if (cell->is_locally_owned())
          {
              Point<dim> p1 = cell->center();
              Point<dim> p2 = cell->center();
              p1[0] -= cell->extent_in_direction(0)*0.5;
              p2[0] += cell->extent_in_direction(0)*0.5;
              p1[1] -= cell->extent_in_direction(1)*0.5;
              p2[1] += cell->extent_in_direction(1)*0.5;
              p1[2] -= cell->extent_in_direction(2)*0.5;
              p2[2] += cell->extent_in_direction(2)*0.5;

              double minval, maxval, mean, err;
              data.compute_min_max(p1, p2, minval, maxval, mean, err);
              if (maxval-minval < 0.1 || (err) < 5000.0)
                cell->clear_refine_flag ();
          }
        }
    }

    template <int dim>
    void
    TomoAMR<dim>::
    declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Mesh refinement");
      {

        prm.enter_subsection("TomoAMR");
        {

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    TomoAMR<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Mesh refinement");
      {
        prm.enter_subsection("TomoAMR");


        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

  }




    template <int dim>
    class TomoInitial : public CompositionalInitialConditions::Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        /**
         * Constructor.
         */

        /**
         * Return the initial composition as a function of position and number
         * of compositional field.
         */
        virtual
        double initial_composition (const Point<dim> &position, const unsigned int n_comp) const
        {
          double minval, maxval, mean, err;
          Point<dim> p = position;
          data.compute_min_max(p, p, minval, maxval, mean, err);
          return (minval+maxval)*0.5;

        }

        /**
         * Declare the parameters this class takes through input files. The
         * default implementation of this function does not describe any
         * parameters. Consequently, derived classes do not have to overload
         * this function if they do not take any runtime parameters.
         */
        static
        void
        declare_parameters (ParameterHandler &prm)
        {}

        /**
         * Read the parameters this class declares from the parameter file.
         * The default implementation of this function does not read any
         * parameters. Consequently, derived classes do not have to overload
         * this function if they do not take any runtime parameters.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm)
        {}

      private:

    };

}

// explicit instantiations
namespace aspect
{
  namespace MeshRefinement
  {
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(TomoAMR,
                                              "TomoAMR",
                                              "TODO")
  }
  ASPECT_REGISTER_COMPOSITIONAL_INITIAL_CONDITIONS(TomoInitial,
                                                   "tomo",
                                                   "")
}
