/*
  Copyright (C) 2020 by the authors of the ASPECT code.

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


#include <aspect/postprocess/sampled_output.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/geometry_model/sphere.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/global.h>
#include <deal.II/numerics/vector_tools.h>

#include <math.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <mpi.h>
#include <netcdf.h>
#include <netcdf_par.h>

namespace
{

  /**
   * POD structure to contain a data value
   * and a priority for MPI communication
   */
  struct DataPriority
  {
    double data;
    double prio;
  };

  // custom MIP_Op for DataPriority
  void
  reduce_op(const void *in_lhs_,
            void       *inout_rhs_,
            int        *len,
            MPI_Datatype *)
  {
    const DataPriority *in_lhs    = static_cast<const DataPriority *>(in_lhs_);
    DataPriority       *inout_rhs = static_cast<DataPriority *>(inout_rhs_);

    for (int i = 0; i < *len; i++)
      {
        if (inout_rhs[i].prio < in_lhs[i].prio)
          {
            inout_rhs[i].data       = in_lhs[i].data;
            inout_rhs[i].prio = in_lhs[i].prio;
          }
      }
  }
} // namespace


namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    SampledOutput<dim>::SampledOutput ()
      :
      // the following value is later read from the input file
      output_interval (0),
      // initialize this to a nonsensical value; set it to the actual time
      // the first time around we get to check it
      last_output_time (std::numeric_limits<double>::quiet_NaN()),
      evaluation_points_cartesian (std::vector<Point<dim> >() ),
      data (
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), // TODO HACK
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
        Utilities::Coordinates::cartesian, 6,
        16, 16, 16),
      use_natural_coordinates (false)
    {}

    template <int dim>
    std::pair<std::string,std::string>
    SampledOutput<dim>::execute (TableHandler &)
    {
      // if this is the first time we get here, set the next output time
      // to the current time. this makes sure we always produce data during
      // the first time step
      if (std::isnan(last_output_time))
        last_output_time = this->get_time() - output_interval;

      // see if output is requested at this time
      if (this->get_time() < last_output_time + output_interval)
        return std::pair<std::string,std::string>();


      // now write all of the data to the file of choice. start with a pre-amble that
      // explains the meaning of the various fields
      static unsigned int output_index = 2;
      ++output_index;



      const int n_ranks = dealii::Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
      const int my_id = dealii::Utilities::MPI::this_mpi_process(this->get_mpi_communicator());




      unsigned int nn = std::pow(2,output_index);
      if (my_id==0)
        std::cout << "structured output as " << nn << "^3" << std::endl;
      data.reinit(nn,nn,nn);

      // create a quadrature formula based on the temperature element alone.
      const QGauss<dim> quadrature_formula (this->get_fe().base_element(this->introspection().base_elements.temperature).degree);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_gradients |
                               update_quadrature_points |
                               update_JxW_values);

      MaterialModel::MaterialModelInputs<dim> in(fe_values.n_quadrature_points, this->n_compositional_fields());
      MaterialModel::MaterialModelOutputs<dim> out(fe_values.n_quadrature_points, this->n_compositional_fields());

      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit (cell);
            in.reinit(fe_values, cell, this->introspection(), this->get_solution());

            this->get_material_model().fill_additional_material_model_inputs(in, this->get_solution(), fe_values, this->introspection());
            this->get_material_model().evaluate(in, out);

            std::vector<double> new_data(6, 0.);

            for (unsigned int q=0; q<n_q_points; ++q)
              {
                new_data[0] = out.viscosities[q];
                new_data[1] = in.temperature[q];
                new_data[2] = in.pressure[q];

                data.splat_data(fe_values.quadrature_point(q),
                                new_data,
                                cell->diameter() /*radius*/);
              }
          }


      data.compress(this->get_mpi_communicator());

      /*
            MPI_Op op;
            int    ierr =
              MPI_Op_create(reinterpret_cast<MPI_User_function *>(&reduce_op),
                            true,
                            &op);
            AssertThrowMPI(ierr);

            ierr = MPI_Allreduce(
              in.data(), result.data(), my_values.size(), type, op, mpi_communicator);
            AssertThrowMPI(ierr);

            ierr = MPI_Type_free(&type);
            AssertThrowMPI(ierr);

            ierr = MPI_Op_free(&op);
            AssertThrowMPI(ierr);
      */


      // TODO: use the radius as priority




      if (false)
        {
          const std::string filename = (this->get_output_directory() +
                                        "sampled_data_"
                                        + Utilities::int_to_string(output_index) +
                                        ".csv");
          std::ofstream f (filename.c_str());
          f << "# ix iy iz c1 c2 radius ";
          f << '\n';


          for (unsigned int ix = 0; ix<data.coordinate_values[0].size(); ++ix)
            for (unsigned int iy = 0; iy<data.coordinate_values[1].size(); ++iy)
              for (unsigned int iz = 0; iz<data.coordinate_values[2].size(); ++iz)
                {
                  TableIndices<dim> idx;
                  idx[0]=ix;
                  idx[1]=iy;
                  idx[2]=iz;
                  f << ix << ' '
                    << iy << ' '
                    << iz << ' '
                    << data.data[0](idx) << ' '
                    << data.data[1](idx) << ' '
                    << data.data[2](idx) << '\n';
                }


          AssertThrow (f, ExcMessage("Writing data to <" + filename +
                                     "> did not succeed in the `point values' "
                                     "postprocessor."));
        }


#ifdef ASPECT_WITH_NETCDF
      {
        const std::string filename = (this->get_output_directory() +
                                      "sampled_data_"
                                      + Utilities::int_to_string(output_index, 4) +
                                      ".nc");
        if (my_id==0)
          std::cout << "writing " << filename
                    << " resolution " << nn << std::endl;

        int ncid;
        int retval;

        retval = nc_create_par(filename.c_str(),
                               NC_CLOBBER | NC_NETCDF4,
                               this->get_mpi_communicator(),
                               MPI_INFO_NULL, // for I/O hints
                               &ncid);
        AssertThrow(retval == 0, ExcMessage("netcdf create failed"));

        int dim_ids[3];
        // note: the last variable varies "fastest", so swap x and z
        if (nc_def_dim(ncid, "X", data.coordinate_values[0].size(), &dim_ids[2]))
          Assert(false, ExcMessage("netcdf command failed"));
        if (nc_def_dim(ncid, "Y", data.coordinate_values[1].size(), &dim_ids[1]))
          Assert(false, ExcMessage("netcdf command failed"));
        if (nc_def_dim(ncid, "Z", data.coordinate_values[2].size(), &dim_ids[0]))
          Assert(false, ExcMessage("netcdf command failed"));

        int varid[6];
        std::array<const char*,6> varnames={"visc", "T", "p", "c4", "myid", "r"};

        for (unsigned int c=0;c<6;++c)
          {
            if (nc_def_var(ncid, varnames[c], NC_FLOAT, 3, dim_ids, &varid[c]))
              Assert(false, ExcMessage("netcdf command failed"));

//            int shuffle = 0;
//              int deflate = 1;
//              int deflate_level = 2;

//              if(nc_def_var_deflate(ncid, varid[c], shuffle, deflate, deflate_level))
//                Assert(false, ExcMessage("netcdf command failed"));

//              if (nc_var_par_access(ncid, varid[c], NC_COLLECTIVE))
//                Assert(false, ExcMessage("netcdf command failed"));

          }

//        if (nc_var_par_access(ncid, NC_GLOBAL, NC_COLLECTIVE))
//          Assert(false, ExcMessage("netcdf command failed"));

        if (nc_enddef(ncid))
          Assert(false, ExcMessage("netcdf command failed"));

        //err = nc_var_par_access(ncid, v1id, NC_COLLECTIVE); ERR

        // hyperslab within global index space:
        size_t startp[3]= {data.my_start_index[2],data.my_start_index[1],data.my_start_index[0]};
        size_t countp[3]=
        {
          data.my_count[2],data.my_count[1],data.my_count[0]
        };

        std::vector<float> structured_data(countp[0]*countp[1]*countp[2]);
        for (unsigned int c=0;c<6;++c)
        {
          float *data_ptr = structured_data.data();
          for (unsigned int iz = 0; iz<countp[0]; ++iz)
            for (unsigned int iy = 0; iy<countp[1]; ++iy)
              for (unsigned int ix = 0; ix<countp[2]; ++ix)
                {
                  TableIndices<dim> idx;
                  idx[0]=ix;
                  idx[1]=iy;
                  idx[2]=iz;
                  *data_ptr = data.data[c](idx);
                  ++data_ptr;
                }

          if (nc_put_vara_float(ncid, varid[c], startp, countp, structured_data.data()))
            Assert(false, ExcMessage("netcdf command failed"));
        }


        {
          float *data_ptr = structured_data.data();
          for (unsigned int iz = 0; iz<countp[2]; ++iz)
            for (unsigned int iy = 0; iy<countp[1]; ++iy)
              for (unsigned int ix = 0; ix<countp[0]; ++ix)
                {
                  TableIndices<dim> idx;
                  idx[0]=ix;
                  idx[1]=iy;
                  idx[2]=iz;
                  *data_ptr = my_id;
                  ++data_ptr;
                }
        }

        if (nc_put_vara_float(ncid, varid[4], startp, countp, structured_data.data()))
          Assert(false, ExcMessage("netcdf command failed"));


        retval = nc_close(ncid);
        AssertThrow(retval == 0, ExcMessage("netcdf close failed"));
      }
#endif

#ifdef DEAL_II_WITH_NETCDFa
      {
        const std::string filename = (this->get_output_directory() +
                                      "sampled_data_"
                                      + Utilities::int_to_string(output_index) +
                                      ".netcdf");



        /* IDs for the netCDF file, dimensions, and variables. */
        int ncid, lon_dimid, lat_dimid, lvl_dimid, rec_dimid;
        int lat_varid, lon_varid, pres_varid, temp_varid;
        int dimids[dim];

        /* The start and count arrays will tell the netCDF library where to
           write our data. */
        size_t start[dim], count[dim];

        /* Program variables to hold the data we will write out. We will only
           need enough space to hold one timestep of data; one record. */
        float pres_out[NLVL][NLAT][NLON];
        float temp_out[NLVL][NLAT][NLON];

        /* These program variables hold the latitudes and longitudes. */
        float lats[NLAT], lons[NLON];

        /* Loop indexes. */
        int lvl, lat, lon, rec, i = 0;

        /* Error handling. */
        int retval;

        /* Create some pretend data. If this wasn't an example program, we
         * would have some real data to write, for example, model
         * output. */
        for (lat = 0; lat < NLAT; lat++)
          lats[lat] = START_LAT + 5.*lat;
        for (lon = 0; lon < NLON; lon++)
          lons[lon] = START_LON + 5.*lon;

        for (lvl = 0; lvl < NLVL; lvl++)
          for (lat = 0; lat < NLAT; lat++)
            for (lon = 0; lon < NLON; lon++)
              {
                pres_out[lvl][lat][lon] = SAMPLE_PRESSURE + i;
                temp_out[lvl][lat][lon] = SAMPLE_TEMP + i++;
              }

        /* Create the file. */
        if ((retval = nc_create(FILE_NAME, NC_CLOBBER, &ncid)))
          ERR(retval);

        /* Define the dimensions. The record dimension is defined to have
         * unlimited length - it can grow as needed. In this example it is
         * the time dimension.*/
        if ((retval = nc_def_dim(ncid, LVL_NAME, NLVL, &lvl_dimid)))
          ERR(retval);
        if ((retval = nc_def_dim(ncid, LAT_NAME, NLAT, &lat_dimid)))
          ERR(retval);
        if ((retval = nc_def_dim(ncid, LON_NAME, NLON, &lon_dimid)))
          ERR(retval);
        if ((retval = nc_def_dim(ncid, REC_NAME, NC_UNLIMITED, &rec_dimid)))
          ERR(retval);

        /* Define the coordinate variables. We will only define coordinate
           variables for lat and lon.  Ordinarily we would need to provide
           an array of dimension IDs for each variable's dimensions, but
           since coordinate variables only have one dimension, we can
           simply provide the address of that dimension ID (&lat_dimid) and
           similarly for (&lon_dimid). */
        if ((retval = nc_def_var(ncid, LAT_NAME, NC_FLOAT, 1, &lat_dimid,
                                 &lat_varid)))
          ERR(retval);
        if ((retval = nc_def_var(ncid, LON_NAME, NC_FLOAT, 1, &lon_dimid,
                                 &lon_varid)))
          ERR(retval);

        /* Assign units attributes to coordinate variables. */
        if ((retval = nc_put_att_text(ncid, lat_varid, UNITS,
                                      strlen(DEGREES_NORTH), DEGREES_NORTH)))
          ERR(retval);
        if ((retval = nc_put_att_text(ncid, lon_varid, UNITS,
                                      strlen(DEGREES_EAST), DEGREES_EAST)))
          ERR(retval);

        /* The dimids array is used to pass the dimids of the dimensions of
           the netCDF variables. Both of the netCDF variables we are
           creating share the same four dimensions. In C, the
           unlimited dimension must come first on the list of dimids. */
        dimids[0] = rec_dimid;
        dimids[1] = lvl_dimid;
        dimids[2] = lat_dimid;
        dimids[3] = lon_dimid;

        /* Define the netCDF variables for the pressure and temperature
         * data. */
        if ((retval = nc_def_var(ncid, PRES_NAME, NC_FLOAT, NDIMS,
                                 dimids, &pres_varid)))
          ERR(retval);
        if ((retval = nc_def_var(ncid, TEMP_NAME, NC_FLOAT, NDIMS,
                                 dimids, &temp_varid)))
          ERR(retval);

        /* Assign units attributes to the netCDF variables. */
        if ((retval = nc_put_att_text(ncid, pres_varid, UNITS,
                                      strlen(PRES_UNITS), PRES_UNITS)))
          ERR(retval);
        if ((retval = nc_put_att_text(ncid, temp_varid, UNITS,
                                      strlen(TEMP_UNITS), TEMP_UNITS)))
          ERR(retval);

        /* End define mode. */
        if ((retval = nc_enddef(ncid)))
          ERR(retval);

        /* Write the coordinate variable data. This will put the latitudes
           and longitudes of our data grid into the netCDF file. */
        if ((retval = nc_put_var_float(ncid, lat_varid, &lats[0])))
          ERR(retval);
        if ((retval = nc_put_var_float(ncid, lon_varid, &lons[0])))
          ERR(retval);

        /* These settings tell netcdf to write one timestep of data. (The
          setting of start[0] inside the loop below tells netCDF which
          timestep to write.) */
        count[0] = 1;
        count[1] = NLVL;
        count[2] = NLAT;
        count[3] = NLON;
        start[1] = 0;
        start[2] = 0;
        start[3] = 0;

        /* Write the pretend data. This will write our surface pressure and
           surface temperature data. The arrays only hold one timestep worth
           of data. We will just rewrite the same data for each timestep. In
           a real application, the data would change between timesteps. */
        for (rec = 0; rec < NREC; rec++)
          {
            start[0] = rec;
            if ((retval = nc_put_vara_float(ncid, pres_varid, start, count,
                                            &pres_out[0][0][0])))
              ERR(retval);
            if ((retval = nc_put_vara_float(ncid, temp_varid, start, count,
                                            &temp_out[0][0][0])))
              ERR(retval);
          }

        /* Close the file. */
        if ((retval = nc_close(ncid)))
          ERR(retval);

      }


#endif



      // Update time
      set_last_output_time (this->get_time());

      // return what should be printed to the screen. note that we had
      // just incremented the number, so use the previous value
      return std::make_pair (std::string ("Writing point values:"),
                             "filename");
    }


    template <int dim>
    void
    SampledOutput<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Sampled Output");
        {
          prm.declare_entry ("Time between point values output", "0.",
                             Patterns::Double (0.),
                             "The time interval between each generation of "
                             "point values output. A value of zero indicates "
                             "that output should be generated in each time step. "
                             "Units: years if the "
                             "'Use years in output instead of seconds' parameter is set; "
                             "seconds otherwise.");
          prm.declare_entry("Evaluation points", "",
                            // a list of points, separated by semicolons; each point has
                            // exactly 'dim' components/coordinates, separated by commas
                            Patterns::List (Patterns::List (Patterns::Double(), dim, dim, ","),
                                            0, Patterns::List::max_int_value, ";"),
                            "The list of points at which the solution should be evaluated. "
                            "Points need to be separated by semicolons, and coordinates of "
                            "each point need to be separated by commas.");
          prm.declare_entry("Use natural coordinates", "false",
                            Patterns::Bool (),
                            "Whether or not the Evaluation points are specified in "
                            "the natural coordinates of the geometry model, e.g. "
                            "radius, lon, lat for the chunk model. "
                            "Currently, natural coordinates for the spherical shell "
                            "and sphere geometries are not supported. ");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    SampledOutput<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Sampled Output");
        {
          output_interval = prm.get_double ("Time between point values output");
          if (this->convert_output_to_years())
            output_interval *= year_in_seconds;

          const std::vector<std::string> point_list
            = Utilities::split_string_list(prm.get("Evaluation points"), ';');

          std::vector<std::array<double,dim> > evaluation_points;

          for (unsigned int p=0; p<point_list.size(); ++p)
            {
              const std::vector<std::string> coordinates
                = Utilities::split_string_list(point_list[p], ',');
              AssertThrow (coordinates.size() == dim,
                           ExcMessage ("In setting up the list of evaluation points for the <Point values> "
                                       "postprocessor, one of the evaluation points reads <"
                                       + point_list[p] +
                                       ">, but this does not correspond to a list of numbers with "
                                       "as many coordinates as you run your simulation in."));

              std::array<double,dim> point;
              for (unsigned int d=0; d<dim; ++d)
                point[d] = Utilities::string_to_double (coordinates[d]);
              evaluation_points.push_back (point);
            }

          use_natural_coordinates = prm.get_bool("Use natural coordinates");

          if (use_natural_coordinates)
            AssertThrow (!Plugins::plugin_type_matches<const GeometryModel::Sphere<dim>>(this->get_geometry_model()) &&
                         !Plugins::plugin_type_matches<const GeometryModel::SphericalShell<dim>>(this->get_geometry_model()),
                         ExcMessage ("This postprocessor can not be used if the geometry "
                                     "is a sphere or spherical shell, because these geometries have not implemented natural coordinates."));

          // Convert the vector of coordinate arrays in Cartesian or natural
          // coordinates to a vector of Point<dim> of Cartesian coordinates.
          evaluation_points_cartesian.resize(evaluation_points.size());
          for (unsigned int p=0; p<evaluation_points.size(); ++p)
            {
              if (use_natural_coordinates)
                evaluation_points_cartesian[p] = this->get_geometry_model().natural_to_cartesian_coordinates(evaluation_points[p]);
              else
                for (unsigned int i = 0; i < dim; i++)
                  evaluation_points_cartesian[p][i] = evaluation_points[p][i];
            }
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    template <class Archive>
    void SampledOutput<dim>::serialize (Archive &ar, const unsigned int)
    {
      ar &evaluation_points_cartesian
      & last_output_time;
    }


    template <int dim>
    void
    SampledOutput<dim>::save (std::map<std::string, std::string> &status_strings) const
    {
      std::ostringstream os;
      aspect::oarchive oa (os);
      oa << (*this);

      status_strings["SampledOutput"] = os.str();
    }


    template <int dim>
    void
    SampledOutput<dim>::load (const std::map<std::string, std::string> &status_strings)
    {
      // see if something was saved
      if (status_strings.find("SampledOutput") != status_strings.end())
        {
          std::istringstream is (status_strings.find("SampledOutput")->second);
          aspect::iarchive ia (is);
          ia >> (*this);
        }
    }


    template <int dim>
    void
    SampledOutput<dim>::set_last_output_time (const double current_time)
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
    ASPECT_REGISTER_POSTPROCESSOR(SampledOutput,
                                  "sampled output",
                                  "A postprocessor that evaluates the solution (i.e., velocity, pressure, "
                                  "temperature, and compositional fields along with other fields that "
                                  "are treated as primary variables) at the end of every time step or "
                                  "after a user-specified time interval "
                                  "at ")
  }
}
