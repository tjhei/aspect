/*
  Copyright (C) 2015 - 2019 by the authors of the ASPECT code.

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

#ifndef _aspect_sampled_h
#define _aspect_sampled_h

#include <aspect/global.h>

// C++11 related includes.
#include <array>
#include <functional>
#include <memory>

// for std_cxx14::make_unique:
#include <deal.II/base/std_cxx14/memory.h>

#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>

#include <deal.II/base/function_lib.h>

#include <aspect/coordinate_systems.h>
#include <aspect/compat.h>

namespace aspect
{

  template <int dim>
  class StructuredData
  {
    public:

      unsigned int n_ranks;
      unsigned int my_id;

      const unsigned int n_components;
      std::array<std::vector<double>, dim> coordinate_values;
      std::vector<Table<dim, double>> data;

      struct SendData
      {
        std::vector<unsigned int> all_indices;
        std::vector<double> all_values;

        void add(const std::array<unsigned int, dim> &index,
                 const std::vector<double> &values,
                 const double priority)
        {
          for (unsigned int d =0; d<dim; ++d)
            all_indices.push_back(index[d]);
          for (unsigned int d =0; d<values.size(); ++d)
            all_values.push_back(values[d]);
          all_values.push_back(priority);
        }
      };
      std::map<unsigned int,SendData> proc_send_data;

      TableIndices<dim> my_start_index;
      TableIndices<dim> my_end_index;
      TableIndices<dim> my_count;

      unsigned int
      find_owner(const TableIndices<dim> &index) const
      {
        // simple algorithm: just split in the first coordinate direction and keep 100%
        // of the other dimensions on a single rank.

        const unsigned int eachx=coordinate_values[0].size()/n_ranks;
        const unsigned int owner_idx = index[0]/eachx;

        // last group can be bigger:
        if (owner_idx>=n_ranks)
          return n_ranks-1;

        Assert(owner_idx <n_ranks, ExcMessage("invalid index in owner()"));
        return owner_idx;
      }

      void
      set_values_local(const TableIndices<dim> &index,
                       std::vector<double> values,
                       double priority)
      {
        Assert(find_owner(index) == my_id,
               ExcMessage("trying to set nonlocal value."));

        TableIndices<dim> localindex;
        for (unsigned int d=0; d<dim; ++d)
          localindex[d] = index[d]-my_start_index[d];

        const double old_1_over_radius = data[n_components](localindex);
        const double new_1_over_radius = priority;
        if (new_1_over_radius > old_1_over_radius)
          {
            data[n_components](localindex) = new_1_over_radius;
            for (unsigned int c=0; c<n_components; ++c)
              data[c](localindex) = values[c];
          }
      }

      void
      set_values(const TableIndices<dim> &index,
                 std::vector<double> values,
                 double priority)
      {
        unsigned int owner = find_owner(index);
        if (owner == my_id)
          set_values_local(index, values, priority);
        else
          {
            auto &dat = proc_send_data[owner];
            std::array<unsigned int, dim> index2;
            for (unsigned int d=0; d<dim; ++d)
              index2[d] = index[d];
            dat.add(index2, values, priority);
          }
      }


      void compress(const MPI_Comm &mpi_comm)
      {
        std::vector<unsigned int> destinations;
        for (auto &it : proc_send_data)
          destinations.emplace_back(it.first);

        const unsigned int n_recv = dealii::Utilities::MPI::compute_n_point_to_point_communications(mpi_comm, destinations);

        std::vector<MPI_Request> requests(proc_send_data.size());
        std::vector<std::vector<char>> data_to_send(proc_send_data.size());

        const int mpi_tag = 123424;

        // send:
        {
          unsigned int idx = 0;
          for (const auto &it : proc_send_data)
            {
              std::vector<char> &buffer = data_to_send[idx];
              buffer.resize(sizeof(size_t));
              dealii::Utilities::pack(it.second.all_indices,
                                      buffer, false);
              size_t shift = buffer.size();
              memcpy(buffer.data(),&shift,sizeof(shift));
              dealii::Utilities::pack(it.second.all_values,
                                      buffer, false);

              const int ierr =
                MPI_Isend(buffer.data(),
                          buffer.size(),
                          MPI_CHAR,
                          it.first,
                          mpi_tag,
                          mpi_comm,
                          &requests[idx]);
              AssertThrowMPI(ierr);

              ++idx;
            }
        }

        // receive:
        {
          std::vector<char> recv_buf;
          for (unsigned int index = 0; index < n_recv; ++index)
            {
              MPI_Status status;
              int ierr = MPI_Probe(MPI_ANY_SOURCE, mpi_tag, mpi_comm, &status);
              AssertThrowMPI(ierr);

              int len;
              ierr = MPI_Get_count(&status, MPI_CHAR, &len);
              AssertThrowMPI(ierr);

              recv_buf.resize(len);
              ierr = MPI_Recv(recv_buf.data(),
                              len,
                              MPI_CHAR,
                              status.MPI_SOURCE,
                              status.MPI_TAG,
                              mpi_comm,
                              &status);
              AssertThrowMPI(ierr);

              size_t shift = 0;
              memcpy(&shift, recv_buf.data(),sizeof(size_t));

              recv_buf.erase(recv_buf.begin(), recv_buf.begin()+sizeof(size_t));
              std::vector<unsigned int> all_indices
                = dealii::Utilities::unpack<std::vector<unsigned int>>(recv_buf, false);
              recv_buf.erase(recv_buf.begin(), recv_buf.begin()+shift-sizeof(size_t));
              std::vector<double> all_values
                = dealii::Utilities::unpack<std::vector<double>>(recv_buf, false);

              unsigned int *index_ptr = all_indices.data();
              double *value_ptr = all_values.data();
              for (unsigned int count = 0; count < all_indices.size()/dim; ++count)
                {
                  TableIndices<dim> index;
                  for (unsigned int d=0; d<dim; ++d)
                    index[d] = index_ptr[d];
                  std::vector<double> values(value_ptr, value_ptr+n_components);
                  const double priority = *(value_ptr+n_components);
                  set_values_local(index, values, priority);

                  index_ptr += dim;
                  value_ptr += n_components+1;
                }
            }
        }



        if (requests.size())
          {
            const int ierr =
              MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            AssertThrowMPI(ierr);
          }

        proc_send_data.clear();
      }

      explicit StructuredData(const unsigned int n_ranks,
                              const unsigned int my_id,
                              const Utilities::Coordinates::CoordinateSystem system,
                              const unsigned int n_components,
                              const unsigned int nx,
                              const unsigned int ny,
                              const unsigned int nz)
        : n_ranks(n_ranks),
          my_id(my_id),
          n_components(n_components)
      {
        reinit(nx,ny,nz);
      }


      void reinit(const unsigned int nx,
                  const unsigned int ny,
                  const unsigned int nz)
      {
        coordinate_values = make_coordinate_values(nx,ny,nz);

        // compute indices:
        {
          for (unsigned int d =0; d<dim; ++d)
            {
              my_start_index[d] = 0;
              my_count[d] = coordinate_values[d].size();
            }

          const unsigned int eachx=coordinate_values[0].size()/n_ranks;
          my_start_index[0] = eachx*my_id;
          if (my_id == n_ranks-1)
            my_count[0] = coordinate_values[0].size()-my_start_index[0];
          else
            my_count[0] = eachx;

          for (unsigned int d =0; d<dim; ++d)
            my_end_index[d] = my_start_index[d] + my_count[d];
        }

        data.clear();
        data.resize(n_components+1, make_a_table());
      }

      void splat_data(const Point<dim> &position,
                      const std::vector<double> &new_data,
                      const double radius)
      {
        Assert(new_data.size() == n_components, ExcMessage(""));

        TableIndices<dim> index = position_to_index(position);
        std::array<unsigned int, dim> extend = approximate_extent(position, radius);

//        unsigned int count = 0;

        for (unsigned int ix = 0; ix<2*extend[0]; ++ix)
          for (unsigned int iy = 0; iy<2*extend[1]; ++iy)
            for (unsigned int iz = 0; iz<2*extend[2]; ++iz)
              {
                TableIndices<dim> current_index;
                current_index[0] = std::max(0l,static_cast<long>(index[0] + ix) - static_cast<long>(extend[0]));
                current_index[1] = std::max(0l,static_cast<long>(index[1] + iy) - static_cast<long>(extend[1]));
                current_index[2] = std::max(0l,static_cast<long>(index[2] + iz) - static_cast<long>(extend[2]));

                if (current_index[0] >= coordinate_values[0].size()
                    ||
                    current_index[1] >= coordinate_values[1].size()
                    ||
                    current_index[2] >= coordinate_values[2].size())
                  continue;

                const double new_1_over_radius = 1./(1e-20+index_to_position(current_index).distance(position));

                //if (find_owner(current_index) == my_id)
                set_values(current_index,new_data,new_1_over_radius);

              }

//        std::cout << "splat with " << count << " updates." << std::endl;

      }



      TableIndices<dim> position_to_index(const Point<dim> &position)
      {
        TableIndices<dim> ix;
        for (unsigned int d = 0; d < dim; ++d)
          {
            // get the index of the first element of the coordinate arrays that is
            // larger than p[d]
            ix[d] = (std::lower_bound(coordinate_values[d].begin(),
                                      coordinate_values[d].end(),
                                      position[d]) -
                     coordinate_values[d].begin());

            // the one we want is the index of the coordinate to the left, however,
            // so decrease it by one (unless we have a point to the left of all, in
            // which case we stay where we are; the formulas below are made in a way
            // that allow us to extend the function by a constant value)
            //
            // to make this work, if we got coordinate_values[d].end(), we actually
            // have to consider the last box which has index size()-2
            if (ix[d] == coordinate_values[d].size())
              ix[d] = coordinate_values[d].size() - 2;
            else if (ix[d] > 0)
              --ix[d];
          }

        return ix;
      }

      const Point<dim> index_to_position(TableIndices<dim> &index)
      {
        Point<dim> result;

        for (unsigned int d = 0; d < dim; ++d)
          result[d] = coordinate_values[d][index[d]];

        return result;
      }

      std::array<unsigned int, dim> approximate_extent(const Point<dim> &position, double radius)
      {
        std::array<unsigned int, dim> result;
        for (unsigned int d=0; d<dim; ++d)
          // TODO: hack
          result[d] = 3;//1+(radius/2.0/(coordinate_values[d][1]-coordinate_values[d][0]));

        return result;
      }

    private:

      static const std::vector< double > make_range(double start, double end, unsigned int N)
      {
        std::vector<double> result(N+1);
        for (unsigned int i=0; i<N+1; ++i)
          {
            result[i] = start + (end-start)*i/N;
          }
        return result;
      }

      static const std::array< std::vector< double >, dim > make_coordinate_values(
          const unsigned int nx,
          const unsigned int ny,
          const unsigned int nz)
      {
        std::array< std::vector< double >, dim > values;
        values[0] = make_range(0.,1.,nx);
        values[1] = make_range(0.,1.,ny);
        values[2] = make_range(0.,1.,nz);

        return values;
      }

      const Table<dim,double> make_a_table() const
      {
        Table<dim,double> table;
        TableIndices<dim> indices;
        for (unsigned int d=0; d<dim; ++d)
          indices[d] = my_count[d];
        table.reinit(indices);
        return table;
      }
  };



}

#endif
