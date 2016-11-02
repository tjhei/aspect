/*
  Copyright (C) 2015 - 2016 by the authors of the ASPECT code.

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

#include <aspect/particle/world.h>
#include <aspect/global.h>
#include <aspect/utilities.h>
#include <aspect/compat.h>
#include <aspect/geometry_model/box.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <boost/serialization/map.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace aspect
{
  namespace Particle
  {
    template <int dim>
    World<dim>::World()
      :
      global_number_of_particles(0),
      next_free_particle_index(0),
      data_offset(numbers::invalid_unsigned_int)
    {}

    template <int dim>
    World<dim>::~World()
    {}

    template <int dim>
    void
    World<dim>::initialize()
    {
      connect_to_signals(this->get_signals());

      if (particle_load_balancing & ParticleLoadBalancing::repartition)
        this->get_triangulation().signals.cell_weight.connect(std_cxx11::bind(&aspect::Particle::World<dim>::cell_weight,
                                                                              std_cxx11::ref(*this),
                                                                              std_cxx11::_1,
                                                                              std_cxx11::_2));
    }

    template <int dim>
    const Property::Manager<dim> &
    World<dim>::get_property_manager() const
    {
      return *property_manager;
    }

    template <int dim>
    const Interpolator::Interface<dim> &
    World<dim>::get_interpolator() const
    {
      return *interpolator;
    }

    template <int dim>
    std::multimap<types::LevelInd, Particle<dim> > &
    World<dim>::get_particles()
    {
      return particles;
    }

    template <int dim>
    const std::multimap<types::LevelInd, Particle<dim> > &
    World<dim>::get_particles() const
    {
      return particles;
    }

    template <int dim>
    const std::multimap<types::LevelInd, Particle<dim> > &
    World<dim>::get_ghost_particles() const
    {
      AssertThrow(update_ghost_particles == true,
                  ExcMessage("A part of the model has requested access to the ghost "
                             "particles, but the parameter 'Update ghost particles' has not "
                             "been set, therefore no ghost particles are available."));
      return ghost_particles;
    }

    template <int dim>
    std::string
    World<dim>::generate_output() const
    {
      // If we do not write output
      // return early with the number of particles that were advected
      if (!output)
        return "";

      TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Output");
      const double output_time = (this->convert_output_to_years() ?
                                  this->get_time() / year_in_seconds :
                                  this->get_time());

      const std::string filename = output->output_particle_data(particles,
                                                                property_manager->get_data_info(),
                                                                output_time);

      return filename;
    }

    template <int dim>
    types::particle_index
    World<dim>::n_global_particles() const
    {
      return global_number_of_particles;
    }

    template <int dim>
    void
    World<dim>::update_n_global_particles()
    {
      global_number_of_particles = dealii::Utilities::MPI::sum (particles.size(), this->get_mpi_communicator());
    }

    template <int dim>
    void
    World<dim>::update_next_free_particle_index()
    {
      types::particle_index locally_highest_index = 0;
      typename std::multimap<types::LevelInd, Particle<dim> >::const_iterator it = particles.begin();
      for (; it!=particles.end(); ++it)
        {
          locally_highest_index = std::max(locally_highest_index,it->second.get_id());
        }

      next_free_particle_index = dealii::Utilities::MPI::max (locally_highest_index, this->get_mpi_communicator()) + 1;
    }

    template <int dim>
    void
    World<dim>::update_global_max_particles_per_cell()
    {
      unsigned int local_max_particles_per_cell(0);
      typename parallel::distributed::Triangulation<dim>::active_cell_iterator cell = this->get_triangulation().begin_active();
      for (; cell!=this->get_triangulation().end(); ++cell)
        if (cell->is_locally_owned())
          {
            const types::LevelInd found_cell = std::make_pair<int, int> (cell->level(),cell->index());
            const unsigned int particles_in_cell = particles.count(found_cell);
            local_max_particles_per_cell = std::max(local_max_particles_per_cell,
                                                    particles_in_cell);
          }

      global_max_particles_per_cell = dealii::Utilities::MPI::max(local_max_particles_per_cell,this->get_mpi_communicator());
    }

    template <int dim>
    void
    World<dim>::connect_to_signals(aspect::SimulatorSignals<dim> &signals)
    {
      signals.post_set_initial_state.connect(std_cxx11::bind(&World<dim>::setup_initial_state,
                                                             std_cxx11::ref(*this)));

      signals.pre_refinement_store_user_data.connect(std_cxx11::bind(&World<dim>::register_store_callback_function,
                                                                     std_cxx11::ref(*this),
                                                                     false,
                                                                     std_cxx11::_1));
      signals.pre_checkpoint_store_user_data.connect(std_cxx11::bind(&World<dim>::register_store_callback_function,
                                                                     std_cxx11::ref(*this),
                                                                     true,
                                                                     std_cxx11::_1));

      signals.post_refinement_load_user_data.connect(std_cxx11::bind(&World<dim>::register_load_callback_function,
                                                                     std_cxx11::ref(*this),
                                                                     false,
                                                                     std_cxx11::_1));
      signals.post_resume_load_user_data.connect(std_cxx11::bind(&World<dim>::register_load_callback_function,
                                                                 std_cxx11::ref(*this),
                                                                 true,
                                                                 std_cxx11::_1));
    }

    template <int dim>
    void
    World<dim>::register_store_callback_function(const bool serialization,
                                                 typename parallel::distributed::Triangulation<dim> &triangulation)
    {
      TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Refine mesh, store");

      // Only save and load tracers if there are any, we might get here for
      // example before the tracer generation in timestep 0, or if somebody
      // selected the tracer postprocessor but generated 0 tracers
      update_global_max_particles_per_cell();

      if (global_max_particles_per_cell > 0)
        {
          const std_cxx11::function<void(const typename parallel::distributed::Triangulation<dim>::cell_iterator &,
                                         const typename parallel::distributed::Triangulation<dim>::CellStatus, void *) > callback_function
            = std_cxx11::bind(&aspect::Particle::World<dim>::store_tracers,
                              std_cxx11::ref(*this),
                              std_cxx11::_1,
                              std_cxx11::_2,
                              std_cxx11::_3);

          // We need to transfer the number of tracers for this cell and
          // the tracer data itself. If we are in the process of refinement
          // (i.e. not in serialization) we need to provide 2^dim times the
          // space for the data in case a cell is coarsened and all tracers
          // of the children have to be stored in the parent cell.
          const std::size_t transfer_size_per_cell = sizeof (unsigned int) +
                                                     (property_manager->get_particle_size() * global_max_particles_per_cell) *
                                                     (serialization ?
                                                      1
                                                      :
                                                      std::pow(2,dim));

          data_offset = triangulation.register_data_attach(transfer_size_per_cell,callback_function);
        }
    }

    template <int dim>
    void
    World<dim>::register_load_callback_function(const bool serialization,
                                                typename parallel::distributed::Triangulation<dim> &triangulation)
    {
      TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Refine mesh, load");

      // All particles have been stored, when we reach this point. Empty the
      // map and fill with new particles.
      particles.clear();

      // If we are resuming from a checkpoint, we first have to register the
      // store function again, to set the triangulation in the same state as
      // before the serialization. Only by this it knows how to deserialize the
      // data correctly. Only do this if something was actually stored.
      if (serialization && (global_max_particles_per_cell > 0))
        {
          const std_cxx11::function<void(const typename parallel::distributed::Triangulation<dim>::cell_iterator &,
                                         const typename parallel::distributed::Triangulation<dim>::CellStatus, void *) > callback_function
            = std_cxx11::bind(&aspect::Particle::World<dim>::store_tracers,
                              std_cxx11::ref(*this),
                              std_cxx11::_1,
                              std_cxx11::_2,
                              std_cxx11::_3);

          // We need to transfer the number of tracers for this cell and
          // the tracer data itself and we need to provide 2^dim times the
          // space for the data in case a cell is coarsened
          const std::size_t transfer_size_per_cell = sizeof (unsigned int) +
                                                     (property_manager->get_particle_size() * global_max_particles_per_cell);
          data_offset = triangulation.register_data_attach(transfer_size_per_cell,callback_function);
        }

      // Check if something was stored and load it
      if (data_offset != numbers::invalid_unsigned_int)
        {
          const std_cxx11::function<void(const typename parallel::distributed::Triangulation<dim>::cell_iterator &,
                                         const typename parallel::distributed::Triangulation<dim>::CellStatus,
                                         const void *) > callback_function
            = std_cxx11::bind(&aspect::Particle::World<dim>::load_tracers,
                              std_cxx11::ref(*this),
                              std_cxx11::_1,
                              std_cxx11::_2,
                              std_cxx11::_3);

          triangulation.notify_ready_to_unpack(data_offset,callback_function);

          apply_particle_per_cell_bounds();

          // Reset offset and update global number of particles. The number
          // can change because of discarded or newly generated particles
          data_offset = numbers::invalid_unsigned_int;
          update_n_global_particles();

          if (update_ghost_particles)
            exchange_ghost_particles();
        }
    }

    template <int dim>
    void
    World<dim>::apply_particle_per_cell_bounds()
    {
      // If any load balancing technique is selected that creates/destroys particles
      if (particle_load_balancing & ParticleLoadBalancing::remove_and_add_particles)
        {
          // First do some preparation for particle generation in poorly
          // populated areas. For this we need to know which particle ids to
          // generate so that they are globally unique.
          // Ensure this by communicating the number of particles that every
          // process is going to generate.
          types::particle_index local_next_particle_index = next_free_particle_index;
          if (particle_load_balancing & ParticleLoadBalancing::add_particles)
            {
              types::particle_index particles_to_add_locally = 0;

              // Loop over all cells and determine the number of particles to generate
              typename DoFHandler<dim>::active_cell_iterator
              cell = this->get_dof_handler().begin_active(),
              endc = this->get_dof_handler().end();

              for (; cell!=endc; ++cell)
                if (cell->is_locally_owned())
                  {
                    const types::LevelInd found_cell(cell->level(),cell->index());
                    const unsigned int particles_in_cell = particles.count(found_cell);

                    if (particles_in_cell < min_particles_per_cell)
                      particles_to_add_locally += static_cast<types::particle_index> (min_particles_per_cell - particles_in_cell);
                  }

              // Determine the starting particle index of this process, which
              // is the highest currently existing particle index plus the sum
              // of the number of newly generated particles of all
              // processes with a lower rank.

              types::particle_index local_start_index = 0.0;
              MPI_Scan(&particles_to_add_locally, &local_start_index, 1, ASPECT_TRACER_INDEX_MPI_TYPE, MPI_SUM, this->get_mpi_communicator());
              local_start_index -= particles_to_add_locally;
              local_next_particle_index += local_start_index;

              const types::particle_index globally_generated_particles =
                dealii::Utilities::MPI::sum(particles_to_add_locally,this->get_mpi_communicator());

              AssertThrow (next_free_particle_index <= std::numeric_limits<types::particle_index>::max() - globally_generated_particles,
                           ExcMessage("There is no free particle index left to generate a new particle id. Please check if your"
                                      "model generates unusually many new particles (by repeatedly deleting and regenerating particles), or"
                                      "recompile deal.II with the DEAL_II_WITH_64BIT_INDICES option enabled, to use 64-bit integers for"
                                      "particle ids."));

              next_free_particle_index += globally_generated_particles;
            }

          boost::mt19937 random_number_generator;

          // Loop over all cells and generate or remove the particles cell-wise
          typename DoFHandler<dim>::active_cell_iterator
          cell = this->get_dof_handler().begin_active(),
          endc = this->get_dof_handler().end();

          for (; cell!=endc; ++cell)
            if (cell->is_locally_owned())
              {
                const types::LevelInd found_cell(cell->level(),cell->index());
                const unsigned int n_particles_in_cell = particles.count(found_cell);

                // Add particles if necessary
                if ((particle_load_balancing & ParticleLoadBalancing::add_particles) &&
                    (n_particles_in_cell < min_particles_per_cell))
                  {
                    for (unsigned int i = n_particles_in_cell; i < min_particles_per_cell; ++i,++local_next_particle_index)
                      {
                        std::pair<aspect::Particle::types::LevelInd,Particle<dim> > new_particle = generator->generate_particle(cell,local_next_particle_index);
                        property_manager->initialize_late_particle(new_particle.second,
                                                                   particles,
                                                                   *interpolator,
                                                                   cell);

                        particles.insert(new_particle);
                      }
                  }

                // Remove particles if necessary
                else if ((particle_load_balancing & ParticleLoadBalancing::remove_particles) &&
                         (n_particles_in_cell > max_particles_per_cell))
                  {
                    const std::pair<typename std::multimap<types::LevelInd, Particle<dim> >::iterator, typename std::multimap<types::LevelInd, Particle<dim> >::iterator>
                    particles_in_cell = particles.equal_range(found_cell);

                    const unsigned int n_particles_to_remove = n_particles_in_cell - max_particles_per_cell;

                    std::set<unsigned int> particle_ids_to_remove;
                    while (particle_ids_to_remove.size() < n_particles_to_remove)
                      particle_ids_to_remove.insert(random_number_generator() % n_particles_in_cell);

                    std::list<typename std::multimap<types::LevelInd, Particle<dim> >::iterator> particles_to_remove;

                    for (std::set<unsigned int>::const_iterator id = particle_ids_to_remove.begin();
                         id != particle_ids_to_remove.end(); ++id)
                      {
                        typename std::multimap<types::LevelInd, Particle<dim> >::iterator particle_to_remove = particles_in_cell.first;
                        std::advance(particle_to_remove,*id);

                        particles_to_remove.push_back(particle_to_remove);
                      }

                    for (typename std::list<typename std::multimap<types::LevelInd, Particle<dim> >::iterator>::iterator particle = particles_to_remove.begin();
                         particle != particles_to_remove.end(); ++particle)
                      {
                        particles.erase(*particle);
                      }
                  }
              }
        }
    }

    template <int dim>
    unsigned int
    World<dim>::cell_weight(const typename parallel::distributed::Triangulation<dim>::cell_iterator &cell,
                            const typename parallel::distributed::Triangulation<dim>::CellStatus status)
    {
      if (cell->active() && !cell->is_locally_owned())
        return 0;

      if (status == parallel::distributed::Triangulation<dim>::CELL_PERSIST
          || status == parallel::distributed::Triangulation<dim>::CELL_REFINE)
        {
          const types::LevelInd found_cell = std::make_pair<int, int> (cell->level(),cell->index());
          const unsigned int n_particles_in_cell = particles.count(found_cell);
          return n_particles_in_cell * tracer_weight;
        }
      else if (status == parallel::distributed::Triangulation<dim>::CELL_COARSEN)
        {
          unsigned int n_particles_in_cell = 0;

          for (unsigned int child_index = 0; child_index < GeometryInfo<dim>::max_children_per_cell; ++child_index)
            {
              const typename parallel::distributed::Triangulation<dim>::cell_iterator child = cell->child(child_index);
              const types::LevelInd found_cell = std::make_pair<int, int> (child->level(),child->index());
              n_particles_in_cell += particles.count(found_cell);
            }
          return n_particles_in_cell * tracer_weight;
        }

      Assert (false, ExcInternalError());
      return 0;
    }

    template <int dim>
    void
    World<dim>::store_tracers(const typename parallel::distributed::Triangulation<dim>::cell_iterator &cell,
                              const typename parallel::distributed::Triangulation<dim>::CellStatus status,
                              void *data)
    {
      unsigned int n_particles_in_cell(0);

      // If the cell persist or is refined store all tracers of the current cell.
      if (status == parallel::distributed::Triangulation<dim>::CELL_PERSIST
          || status == parallel::distributed::Triangulation<dim>::CELL_REFINE)
        {
          const types::LevelInd found_cell = std::make_pair<int, int> (cell->level(),cell->index());
          const std::pair<typename std::multimap<types::LevelInd, Particle<dim> >::iterator, typename std::multimap<types::LevelInd, Particle<dim> >::iterator> particles_in_cell
            = particles.equal_range(found_cell);
          n_particles_in_cell = std::distance(particles_in_cell.first,particles_in_cell.second);

          unsigned int *ndata = static_cast<unsigned int *> (data);
          *ndata = n_particles_in_cell;
          data = static_cast<void *> (ndata + 1);

          for (typename std::multimap<types::LevelInd, Particle<dim> >::iterator particle = particles_in_cell.first;
               particle != particles_in_cell.second; ++particle)
            {
              particle->second.write_data(data);
            }
        }
      // If this cell is the parent of children that will be coarsened, collect
      // the tracers of all children.
      // First check if the maximum number of particles per cell is exceeded for
      // the new cell, and if that is the case, only store every 2^dim 'th
      // particle.
      else if (status == parallel::distributed::Triangulation<dim>::CELL_COARSEN)
        {
          for (unsigned int child_index = 0; child_index < GeometryInfo<dim>::max_children_per_cell; ++child_index)
            {
              const typename parallel::distributed::Triangulation<dim>::cell_iterator child = cell->child(child_index);
              const types::LevelInd found_cell = std::make_pair<int, int> (child->level(),child->index());
              const std::pair<typename std::multimap<types::LevelInd, Particle<dim> >::iterator, typename std::multimap<types::LevelInd, Particle<dim> >::iterator> particles_in_cell
                = particles.equal_range(found_cell);
              n_particles_in_cell += std::distance(particles_in_cell.first,particles_in_cell.second);
            }

          unsigned int *ndata = static_cast<unsigned int *> (data);
          *ndata = n_particles_in_cell;

          data = static_cast<void *> (ndata + 1);

          for (unsigned int child_index = 0; child_index < GeometryInfo<dim>::max_children_per_cell; ++child_index)
            {
              const typename parallel::distributed::Triangulation<dim>::cell_iterator child = cell->child(child_index);
              const types::LevelInd found_cell = std::make_pair<int, int> (child->level(),child->index());
              const std::pair<typename std::multimap<types::LevelInd, Particle<dim> >::iterator, typename std::multimap<types::LevelInd, Particle<dim> >::iterator>
              particles_in_cell = particles.equal_range(found_cell);

              for (typename std::multimap<types::LevelInd, Particle<dim> >::iterator particle = particles_in_cell.first;
                   particle != particles_in_cell.second; ++particle)
                {
                  particle->second.write_data(data);
                }
            }
        }
      else
        Assert (false, ExcInternalError());

    }

    template <int dim>
    void
    World<dim>::load_tracers(const typename parallel::distributed::Triangulation<dim>::cell_iterator &cell,
                             const typename parallel::distributed::Triangulation<dim>::CellStatus status,
                             const void *data)
    {
      const unsigned int *n_particles_in_cell_ptr = static_cast<const unsigned int *> (data);
      const void *pdata = reinterpret_cast<const void *> (n_particles_in_cell_ptr + 1);

      if (*n_particles_in_cell_ptr == 0)
        return;

      // Load all particles from the data stream and store them in the local
      // particle map.
      if (status == parallel::distributed::Triangulation<dim>::CELL_PERSIST)
        {
          typename std::multimap<types::LevelInd,Particle<dim> >::const_iterator position_hint = particles.end();
          for (unsigned int i = 0; i < *n_particles_in_cell_ptr; ++i)
            {
#ifdef DEAL_II_WITH_CXX14
              position_hint = particles.emplace_hint(position_hint,
                                                     std::make_pair(cell->level(),cell->index()),
                                                     Particle<dim>(pdata,property_manager->get_particle_size()));
#else
              position_hint = particles.insert(position_hint,
                                               std::make_pair(std::make_pair(cell->level(),cell->index()),
                                                              Particle<dim>(pdata,property_manager->get_particle_size())));
#endif
              ++position_hint;
            }
        }

      else if (status == parallel::distributed::Triangulation<dim>::CELL_COARSEN)
        {
          typename std::multimap<types::LevelInd,Particle<dim> >::iterator position_hint = particles.end();
          for (unsigned int i = 0; i < *n_particles_in_cell_ptr; ++i)
            {
#ifdef DEAL_II_WITH_CXX14
              position_hint = particles.emplace_hint(position_hint,
                                                     std::make_pair(cell->level(),cell->index()),
                                                     Particle<dim>(pdata,property_manager->get_particle_size()));
#else
              position_hint = particles.insert(position_hint,
                                               std::make_pair(std::make_pair(cell->level(),cell->index()),
                                                              Particle<dim>(pdata,property_manager->get_particle_size())));
#endif
              const Point<dim> p_unit = this->get_mapping().transform_real_to_unit_cell(cell, position_hint->second.get_location());
              position_hint->second.set_reference_location(p_unit);
              ++position_hint;
            }
        }
      else if (status == parallel::distributed::Triangulation<dim>::CELL_REFINE)
        {
          std::vector<typename std::multimap<types::LevelInd, Particle<dim> >::const_iterator > position_hints(GeometryInfo<dim>::max_children_per_cell);
          for (unsigned int child_index=0; child_index<GeometryInfo<dim>::max_children_per_cell; ++child_index)
            {
              const typename parallel::distributed::Triangulation<dim>::cell_iterator child = cell->child(child_index);
              position_hints[child_index] = particles.upper_bound(std::make_pair(child->level(),child->index()));
            }

          for (unsigned int i = 0; i < *n_particles_in_cell_ptr; ++i)
            {
              Particle<dim> p (pdata,property_manager->get_particle_size());

              for (unsigned int child_index = 0; child_index < GeometryInfo<dim>::max_children_per_cell; ++child_index)
                {
                  const typename parallel::distributed::Triangulation<dim>::cell_iterator child = cell->child(child_index);

                  try
                    {
                      const Point<dim> p_unit = this->get_mapping().transform_real_to_unit_cell(child,
                                                                                                p.get_location());
                      if (GeometryInfo<dim>::is_inside_unit_cell(p_unit))
                        {
                          p.set_reference_location(p_unit);
#ifdef DEAL_II_WITH_CXX14
                          position_hints[child_index] = particles.emplace_hint(position_hints[child_index],
                                                                               std::make_pair(child->level(),child->index()),
                                                                               std::move(p));
#else
                          position_hints[child_index] = particles.insert(position_hints[child_index],
                                                                         std::make_pair(std::make_pair(child->level(),child->index()),
                                                                                        p));
#endif
                          ++position_hints[child_index];
                          break;
                        }
                    }
                  catch (typename Mapping<dim>::ExcTransformationFailed &)
                    {}
                }
            }
        }
    }

    template <int dim>
    void
    World<dim>::exchange_ghost_particles()
    {
      TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Exchange ghosts");

      // First clear the current ghost_particle information
      ghost_particles.clear();

      std::multimap<types::subdomain_id, std::pair<types::LevelInd, Particle<dim> > > ghost_particles_by_domain;
      std::vector<std::set<unsigned int> > vertex_to_neighbor_subdomain(this->get_triangulation().n_vertices());

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();
      for (; cell != endc; ++cell)
        {
          if (cell->is_ghost())
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
              vertex_to_neighbor_subdomain[cell->vertex_index(v)].insert(cell->subdomain_id());
        }

      cell = this->get_triangulation().begin_active();
      for (; cell != endc; ++cell)
        {
          if (!cell->is_ghost())
            {
              std::set<unsigned int> cell_to_neighbor_subdomain;
              for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                {
                  cell_to_neighbor_subdomain.insert(vertex_to_neighbor_subdomain[cell->vertex_index(v)].begin(),
                                                    vertex_to_neighbor_subdomain[cell->vertex_index(v)].end());
                }

              if (cell_to_neighbor_subdomain.size() > 0)
                {
                  const std::pair< const typename std::multimap<types::LevelInd,Particle <dim> >::iterator,
                        const typename std::multimap<types::LevelInd,Particle <dim> >::iterator>
                        particle_range_in_cell = particles.equal_range(std::make_pair(cell->level(),cell->index()));

                  for (std::set<unsigned int>::iterator domain=cell_to_neighbor_subdomain.begin();
                       domain != cell_to_neighbor_subdomain.end(); ++domain)
                    {
                      for (typename std::multimap<types::LevelInd,Particle <dim> >::iterator particle = particle_range_in_cell.first;
                           particle != particle_range_in_cell.second;
                           ++particle)

                        ghost_particles_by_domain.insert(std::make_pair(*domain,*particle));
                    }
                }
            }
        }

      std::vector<std::pair<types::LevelInd, Particle<dim> > > received_ghost_particles;
      send_recv_particles(ghost_particles_by_domain,
                          received_ghost_particles);

      ghost_particles.insert(received_ghost_particles.begin(),
                             received_ghost_particles.end());
    }

    template <int dim>
    std::multimap<double, typename parallel::distributed::Triangulation<dim>::active_cell_iterator>
    World<dim>::neighbor_cells_to_search(const Particle<dim> &particle,
                                         const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const
    {
      std::multimap<double,typename parallel::distributed::Triangulation<dim>::active_cell_iterator> neighbor_cells;
      for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
        if (cell->at_boundary(face_no) == false)
          {
            if (cell->neighbor(face_no)->active())
              {
                const double center_distance = (particle.get_location() - cell->face(face_no)->center()).norm();
                neighbor_cells.insert(std::make_pair(center_distance,cell->neighbor(face_no)));
              }
            else
              for (unsigned int subface_no=0; subface_no<GeometryInfo<dim>::max_children_per_face; ++subface_no)
                {
                  const typename parallel::distributed::Triangulation<dim>::active_cell_iterator child = cell->neighbor_child_on_subface(face_no,subface_no);
                  const double center_distance = (particle.get_location() - child->face(cell->neighbor_of_neighbor(face_no))->center()).norm();
                  neighbor_cells.insert(std::make_pair(center_distance,child));
                }
          }

      return neighbor_cells;
    }

    template <int dim>
    void
    World<dim>::sort_particles_in_subdomains_and_cells(const std::vector<std::pair<types::LevelInd, Particle<dim> > > &particles_to_sort)
    {
      // TODO: The current algorithm only works for CFL numbers <= 1.0,
      // because it only knows the subdomain_id of ghost cells, but not
      // of artificial cells.

      // There are three reasons why a particle is not in its old cell:
      // It moved to another cell, to another subdomain or it left the mesh.
      // Particles that moved to another cell are updated and stored inside the
      // sorted_particles vector, particles that moved to another domain are
      // collected in the moved_particles_domain multimap. Particles that left
      // the mesh completely are collected in the lost_particles vector.
      std::vector<std::pair<types::LevelInd, Particle<dim> > > sorted_particles;
      std::multimap<types::subdomain_id, std::pair<types::LevelInd, Particle<dim> > > moved_particles_domain;
      std::vector<std::pair<types::LevelInd, Particle<dim> > > lost_particles;

      // We do not know exactly how many particles are lost, exchanged between
      // domains, or remain on this process. Therefore we pre-allocate approximate
      // sizes for these vectors. If more space is needed an automatic and
      // relatively fast (compared to other parts of this algorithm)
      // re-allocation will happen.
      sorted_particles.reserve(static_cast<unsigned int> (particles_to_sort.size()*1.25));
      lost_particles.reserve(static_cast<unsigned int> (particles_to_sort.size()*0.25));

      {
        TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Sort");

        // Find the cells that the particles moved to.
        typename std::vector<std::pair<types::LevelInd, Particle<dim> > >::const_iterator   it;
        for (it=particles_to_sort.begin(); it!=particles_to_sort.end(); ++it)
          {
            // The cell the particle is in
            typename parallel::distributed::Triangulation<dim>::active_cell_iterator current_cell;
            Point<dim> current_reference_position;
            bool found_cell = false;

            // If we know the particle's old cell, check if it is still inside
            // or in one of its neighbors
            if (it->first != std::make_pair(-1,-1))
              {
                current_cell = typename parallel::distributed::Triangulation<dim>::active_cell_iterator (&(this->get_triangulation()), it->first.first, it->first.second);

                // Search all of the neighbors of the previous cell
                // Most likely we will find the particle in them.
                const std::multimap<double, typename parallel::distributed::Triangulation<dim>::active_cell_iterator>
                neighbor_cells = neighbor_cells_to_search(it->second,current_cell);

                for (typename std::multimap<double, typename parallel::distributed::Triangulation<dim>::active_cell_iterator>::const_iterator neighbor_cell = neighbor_cells.begin();
                     neighbor_cell != neighbor_cells.end(); ++neighbor_cell)
                  {
                    try
                      {
                        const Point<dim> p_unit = this->get_mapping().transform_real_to_unit_cell(neighbor_cell->second, it->second.get_location());
                        if (GeometryInfo<dim>::is_inside_unit_cell(p_unit))
                          {
                            current_cell = neighbor_cell->second;
                            current_reference_position = p_unit;
                            found_cell = true;
                            break;
                          }
                      }
                    catch (typename Mapping<dim>::ExcTransformationFailed &)
                      {}
                  }
              }

            if (!found_cell)
              {
                // The particle is not in a neighbor of the old cell.
                // Look for the new cell in the whole local domain.
                // This case is rare.
                try
                  {
                    const std::pair<const typename parallel::distributed::Triangulation<dim>::active_cell_iterator,
                          Point<dim> > current_cell_and_position =
                            GridTools::find_active_cell_around_point<> (this->get_mapping(),
                                                                        this->get_triangulation(),
                                                                        it->second.get_location());
                    current_cell = current_cell_and_position.first;
                    current_reference_position = current_cell_and_position.second;
                  }
                catch (GridTools::ExcPointNotFound<dim> &)
                  {
                    // We can find no cell for this particle. It has left the
                    // domain due to an integration error or an open boundary.
                    lost_particles.push_back(*it);
                    continue;
                  }
              }

            std::pair<types::LevelInd, Particle<dim> > cell_and_particle = *it;
            cell_and_particle.first = std::make_pair(current_cell->level(),current_cell->index());
            cell_and_particle.second.set_reference_location(current_reference_position);

            // Reinsert the particle into our domain if we own its cell.
            // Mark it for MPI transfer otherwise
            if (current_cell->is_locally_owned())
              sorted_particles.push_back(cell_and_particle);
            else
              moved_particles_domain.insert(std::make_pair(current_cell->subdomain_id(),cell_and_particle));
          }

        // If particles fell out of the mesh, put them back in if they have crossed
        // a periodic boundary. If they have left the mesh otherwise, they will be
        // discarded by being deleted from lost_particles, and not inserted anywhere.
        move_particles_back_into_mesh(lost_particles,
                                      sorted_particles,
                                      moved_particles_domain);

      }
      // Exchange particles between processors if we have more than one process
      if (dealii::Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) > 1)
        {
          TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Communicate");
          send_recv_particles(moved_particles_domain,sorted_particles);
        }

      // Sort the updated particles. This pre-sort speeds up inserting
      // them into particles to O(N) complexity.
      const std::multimap<types::LevelInd,Particle <dim> > sorted_particles_map(sorted_particles.begin(),
                                                                                sorted_particles.end());

      particles.insert(sorted_particles_map.begin(),sorted_particles_map.end());
    }

    template <int dim>
    void
    World<dim>::move_particles_back_into_mesh(const std::vector<std::pair<types::LevelInd, Particle<dim> > >                  &lost_particles,
                                              std::vector<std::pair<types::LevelInd, Particle<dim> > >                        &moved_particles_cell,
                                              std::multimap<types::subdomain_id, std::pair<types::LevelInd,Particle<dim> > >  &moved_particles_domain)
    {
      // TODO: fix this to work with arbitrary meshes. Currently periodic boundaries only work for boxes.
      // If the geometry is not a box, we simply discard particles that have left the
      // model domain.

      const GeometryModel::Box<dim> *geometry
        = dynamic_cast<const GeometryModel::Box<dim>*> (&this->get_geometry_model());

      if (geometry != 0)
        {
          const Point<dim> origin = geometry->get_origin();
          const Point<dim> extent = geometry->get_extents();
          const std::set< std::pair< std::pair<types::boundary_id, types::boundary_id>, unsigned int> > periodic_boundaries =
            geometry->get_periodic_boundary_pairs();

          std::vector<bool> periodic(dim,false);
          std::set< std::pair< std::pair<types::boundary_id, types::boundary_id>, unsigned int> >::const_iterator boundary =
            periodic_boundaries.begin();
          for (; boundary != periodic_boundaries.end(); ++boundary)
            periodic[boundary->second] = true;

          typename std::vector<std::pair<types::LevelInd, Particle<dim> > >::const_iterator lost_particle = lost_particles.begin();
          for (; lost_particle != lost_particles.end(); ++lost_particle)
            {
              // modify the particle position if it crossed a periodic boundary
              std::pair<types::LevelInd, Particle<dim> > cell_and_particle = *lost_particle;
              Point<dim> particle_position = cell_and_particle.second.get_location();
              for (unsigned int i = 0; i < dim; ++i)
                {
                  if (periodic[i])
                    {
                      if (particle_position[i] < origin[i])
                        particle_position[i] += extent[i];
                      else if (particle_position[i] > origin[i] + extent[i])
                        particle_position[i] -= extent[i];
                    }
                }
              cell_and_particle.second.set_location(particle_position);

              // Try again looking for the new cell with the updated position
              typename parallel::distributed::Triangulation<dim>::active_cell_iterator cell;
              try
                {
                  const std::pair<const typename parallel::distributed::Triangulation<dim>::active_cell_iterator,
                        Point<dim> > current_cell_and_position =
                          GridTools::find_active_cell_around_point<> (this->get_mapping(),
                                                                      this->get_triangulation(),
                                                                      cell_and_particle.second.get_location());
                  cell = current_cell_and_position.first;
                  cell_and_particle.first = std::make_pair(cell->level(),cell->index());
                  cell_and_particle.second.set_reference_location(current_cell_and_position.second);
                }
              catch (GridTools::ExcPointNotFound<dim> &)
                {
                  // If we can find no cell for this particle there is no hope left
                  // to find its cell. Simply delete the particle.
                  continue;
                }

              // Reinsert the particle into our domain if we found its cell
              // Mark it for MPI transfer otherwise
              if (cell->is_locally_owned())
                moved_particles_cell.push_back(cell_and_particle);
              else
                moved_particles_domain.insert(std::make_pair(cell->subdomain_id(),cell_and_particle));
            }
        }
    }

    template <int dim>
    void
    World<dim>::send_recv_particles(const std::multimap<types::subdomain_id, std::pair<types::LevelInd,Particle <dim> > > &send_particles,
                                    std::vector<std::pair<types::LevelInd, Particle<dim> > >                              &received_particles)
    {
      // Determine the communication pattern
      const std::vector<types::subdomain_id> neighbors (this->get_triangulation().ghost_owners().begin(),
                                                        this->get_triangulation().ghost_owners().end());
      const unsigned int n_neighbors = neighbors.size();

#if DEAL_II_VERSION_GTE(8,5,0)
      // The amount of data per cell id depends on coarse cell and level,
      // assume the worst and allocate enough memory for the largest possible
      // CellIds.
      const CellId test_cid(this->get_triangulation().end(0)->index(),
                            std::vector<unsigned char>(this->get_triangulation().n_global_levels(),'0'));
      const unsigned int max_cellid_size = test_cid.to_string().size();

      const unsigned int particle_size = property_manager->get_particle_size() + integrator->get_data_size() + max_cellid_size;
#else
      const unsigned int particle_size = property_manager->get_particle_size() + integrator->get_data_size();
#endif

      // Determine the amount of data we will send to other processors
      std::vector<unsigned int> n_send_data(n_neighbors);
      std::vector<unsigned int> n_recv_data(n_neighbors);

      std::vector<unsigned int> send_offsets(n_neighbors);
      std::vector<unsigned int> recv_offsets(n_neighbors);

      // Allocate space for sending and receiving particle data
      std::vector<char> send_data(send_particles.size() * particle_size);
      void *data = static_cast<void *> (&send_data.front());

      for (types::subdomain_id neighbor_id = 0; neighbor_id < n_neighbors; ++neighbor_id)
        {
          send_offsets[neighbor_id] = reinterpret_cast<std::size_t> (data) - reinterpret_cast<std::size_t> (&send_data.front());

          const typename std::pair< const typename std::multimap<types::subdomain_id, std::pair<types::LevelInd,Particle <dim> > >::const_iterator,
                const typename std::multimap<types::subdomain_id, std::pair<types::LevelInd,Particle <dim> > >::const_iterator>
                send_particle_range = send_particles.equal_range(neighbors[neighbor_id]);

          // Copy the particle data into the send array
          typename std::multimap<types::subdomain_id,std::pair<types::LevelInd,Particle <dim> > >::const_iterator cell_particle = send_particle_range.first;
          for (; cell_particle != send_particle_range.second; ++cell_particle)
            {
#if DEAL_II_VERSION_GTE(8,5,0)
              const typename parallel::distributed::Triangulation<dim>::cell_iterator cell (&this->get_triangulation(),
                                                                                            cell_particle->second.first.first,
                                                                                            cell_particle->second.first.second);
              const std::string cellid = cell->id().to_string();
              cellid.copy(static_cast<char *>(data),cellid.size());
              data = static_cast<char *>(data) + cellid.size();
#endif
              cell_particle->second.second.write_data(data);
              data = integrator->write_data(data, cell_particle->second.second.get_id());
            }
          n_send_data[neighbor_id] = reinterpret_cast<std::size_t> (data) - send_offsets[neighbor_id] - reinterpret_cast<std::size_t> (&send_data.front());
        }

      // Notify other processors how many particles we will send
      std::vector<MPI_Request> n_requests(2*n_neighbors);
      for (unsigned int i=0; i<n_neighbors; ++i)
        MPI_Irecv(&(n_recv_data[i]), 1, MPI_INT, neighbors[i], 0, this->get_mpi_communicator(), &(n_requests[2*i]));
      for (unsigned int i=0; i<n_neighbors; ++i)
        MPI_Isend(&(n_send_data[i]), 1, MPI_INT, neighbors[i], 0, this->get_mpi_communicator(), &(n_requests[2*i+1]));
      MPI_Waitall(2*n_neighbors,&n_requests[0],MPI_STATUSES_IGNORE);

      // Determine how many particles and data we will receive
      unsigned int total_recv_data = 0;
      for (unsigned int neighbor_id=0; neighbor_id<n_neighbors; ++neighbor_id)
        {
          recv_offsets[neighbor_id] = total_recv_data;
          total_recv_data += n_recv_data[neighbor_id];
        }

      // Set up the space for the received particle data
      std::vector<char> recv_data(total_recv_data);

      // Exchange the particle data between domains
      std::vector<MPI_Request> requests(2*n_neighbors);
      unsigned int send_ops = 0;
      unsigned int recv_ops = 0;

      for (unsigned int i=0; i<n_neighbors; ++i)
        if (n_recv_data[i] > 0)
          {
            MPI_Irecv(&(recv_data[recv_offsets[i]]), n_recv_data[i], MPI_CHAR, neighbors[i], 1, this->get_mpi_communicator(),&(requests[send_ops]));
            send_ops++;
          }

      for (unsigned int i=0; i<n_neighbors; ++i)
        if (n_send_data[i] > 0)
          {
            MPI_Isend(&(send_data[send_offsets[i]]), n_send_data[i], MPI_CHAR, neighbors[i], 1, this->get_mpi_communicator(),&(requests[send_ops+recv_ops]));
            recv_ops++;
          }
      MPI_Waitall(send_ops+recv_ops,&requests[0],MPI_STATUSES_IGNORE);

      // Put the received particles into the domain if they are in the triangulation
      const void *recv_data_it = static_cast<const void *> (&recv_data.front());

      while (reinterpret_cast<std::size_t> (recv_data_it) - reinterpret_cast<std::size_t> (&recv_data.front()) < total_recv_data)
        {
#if DEAL_II_VERSION_GTE(8,5,0)
          const std::string cell_string(reinterpret_cast<const char *> (recv_data_it),max_cellid_size);
          std::stringstream stream(cell_string);
          CellId id;
          stream >> id;
          recv_data_it = static_cast<const char *> (recv_data_it) + id.to_string().size();

          const typename parallel::distributed::Triangulation<dim>::active_cell_iterator cell = id.to_cell(this->get_triangulation());
          const Particle<dim> recv_particle(recv_data_it,property_manager->get_particle_size());
          recv_data_it = integrator->read_data(recv_data_it, recv_particle.get_id());
#else
          Particle<dim> recv_particle(recv_data_it,property_manager->get_particle_size());
          recv_data_it = integrator->read_data(recv_data_it, recv_particle.get_id());
          const std::pair<const typename parallel::distributed::Triangulation<dim>::active_cell_iterator,
                Point<dim> > current_cell_and_position =
                  GridTools::find_active_cell_around_point<> (this->get_mapping(),
                                                              this->get_triangulation(),
                                                              recv_particle.get_location());
          typename parallel::distributed::Triangulation<dim>::active_cell_iterator cell = current_cell_and_position.first;
          recv_particle.set_reference_location(current_cell_and_position.second);

          // GridTools::find_active_cell_around_point can find a different cell than
          // particle_is_in_cell if the particle is very close to the boundary
          // therefore, we might get a cell here that does not belong to us.
          // But then at least one of its neighbors belongs to us, and the particle
          // is extremely close to the boundary of these two cells. Look in the
          // neighbor cells for the particle.
          if (!cell->is_locally_owned())
            {
              // Now try again for all of the neighbors of the cell
              // Most likely we will find the particle in them.
              const std::multimap<double, typename parallel::distributed::Triangulation<dim>::active_cell_iterator>
              neighbor_cells = neighbor_cells_to_search(recv_particle,cell);

              for (typename std::map<double, typename parallel::distributed::Triangulation<dim>::active_cell_iterator>::const_iterator neighbor_cell = neighbor_cells.begin();
                   neighbor_cell != neighbor_cells.end(); ++neighbor_cell)
                {
                  try
                    {
                      const Point<dim> p_unit = this->get_mapping().transform_real_to_unit_cell(neighbor_cell->second, recv_particle.get_location());
                      if (GeometryInfo<dim>::is_inside_unit_cell(p_unit))
                        {
                          cell = neighbor_cell->second;
                          recv_particle.set_reference_location(p_unit);
                          break;
                        }
                      // If the particle is not in this cell, do nothing and check
                      // the next neighbor cell
                    }
                  catch (typename Mapping<dim>::ExcTransformationFailed &)
                    {}
                }
            }
#endif

          const types::LevelInd found_cell = std::make_pair(cell->level(),cell->index());

          received_particles.push_back(std::make_pair(found_cell, recv_particle));
        }

      AssertThrow(recv_data_it == &recv_data.back()+1,
                  ExcMessage("The amount of data that was read into new particles "
                             "does not match the amount of data sent around."));
    }

    template <int dim>
    void
    World<dim>::local_initialize_particles(const typename std::multimap<types::LevelInd, Particle<dim> >::iterator &begin_particle,
                                           const typename std::multimap<types::LevelInd, Particle<dim> >::iterator &end_particle)
    {
      for (typename std::multimap<types::LevelInd, Particle<dim> >::iterator it = begin_particle; it!=end_particle; ++it)
        property_manager->initialize_one_particle(it->second);
    }

    template <int dim>
    void
    World<dim>::local_update_particles(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                       const typename std::multimap<types::LevelInd, Particle<dim> >::iterator &begin_particle,
                                       const typename std::multimap<types::LevelInd, Particle<dim> >::iterator &end_particle)
    {
      const unsigned int particles_in_cell = std::distance(begin_particle,end_particle);
      const unsigned int solution_components = this->introspection().n_components;

      Vector<double>              value (solution_components);
      std::vector<Tensor<1,dim> > gradient (solution_components,Tensor<1,dim>());

      std::vector<Vector<double> >              values(particles_in_cell,value);
      std::vector<std::vector<Tensor<1,dim> > > gradients(particles_in_cell,gradient);
      std::vector<Point<dim> >                  positions(particles_in_cell);

      typename std::multimap<types::LevelInd, Particle<dim> >::iterator it = begin_particle;
      for (unsigned int i = 0; it!=end_particle; ++it,++i)
        {
          positions[i] = it->second.get_reference_location();
        }

      const Quadrature<dim> quadrature_formula(positions);
      const UpdateFlags update_flags = property_manager->get_needed_update_flags();
      FEValues<dim> fe_value (this->get_mapping(),
                              this->get_fe(),
                              quadrature_formula,
                              update_flags);

      fe_value.reinit (cell);
      if (update_flags & update_values)
        fe_value.get_function_values (this->get_solution(),
                                      values);
      if (update_flags & update_gradients)
        fe_value.get_function_gradients (this->get_solution(),
                                         gradients);

      it = begin_particle;
      for (unsigned int i = 0; it!=end_particle; ++it,++i)
        {
          property_manager->update_one_particle(it->second,
                                                values[i],
                                                gradients[i]);
        }
    }

    template <int dim>
    void
    World<dim>::local_advect_particles(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                       const typename std::multimap<types::LevelInd, Particle<dim> >::iterator &begin_particle,
                                       const typename std::multimap<types::LevelInd, Particle<dim> >::iterator &end_particle,
                                       std::vector<std::pair<types::LevelInd, Particle <dim> > > &particles_out_of_cell)
    {
      const unsigned int particles_in_cell = std::distance(begin_particle,end_particle);

      std::vector<Tensor<1,dim> >  result(particles_in_cell);
      std::vector<Tensor<1,dim> >  old_result(particles_in_cell);

      // Below we manually evaluate the solution at all support points of the
      // current cell, and then use the shape functions to interpolate the
      // solution to the particle points. All of this can be done with less
      // code using an FEValues object, but since this object initializes a lot
      // of memory for other purposes and we can not reuse the FEValues object
      // for other cells, it is much faster to do the work manually. Also this
      // function is quite performance critical.

      std::vector<types::global_dof_index> cell_dof_indices (this->get_fe().dofs_per_cell);
      cell->get_dof_indices (cell_dof_indices);

      const FiniteElement<dim> &velocity_fe = this->get_fe().base_element(this->introspection().base_elements.velocities);

      for (unsigned int j=0; j<velocity_fe.dofs_per_cell; ++j)
        {
          Tensor<1,dim> solution_at_support_point;
          Tensor<1,dim> old_solution_at_support_point;
          for (unsigned int dir=0; dir<dim; ++dir)
            {
              const unsigned int support_point_index
                = this->get_fe().component_to_system_index(/*velocity component=*/ this->introspection().component_indices.velocities[dir],
                                                                                   /*dof index within component=*/ j);
              solution_at_support_point[dir] = this->get_solution()[cell_dof_indices[support_point_index]];
              old_solution_at_support_point[dir] = this->get_old_solution()[cell_dof_indices[support_point_index]];
            }

          typename std::multimap<types::LevelInd, Particle<dim> >::iterator it = begin_particle;
          for (unsigned int particle_index = 0; it!=end_particle; ++it,++particle_index)
            {
              const double shape_value = velocity_fe.shape_value(j,it->second.get_reference_location());
              result[particle_index] += solution_at_support_point * shape_value;
              old_result[particle_index] += old_solution_at_support_point * shape_value;
            }
        }

      integrator->local_integrate_step(begin_particle,
                                       end_particle,
                                       old_result,
                                       result,
                                       this->get_timestep());

      // Now update the reference locations of the moved particles
      for (typename std::multimap<types::LevelInd, Particle<dim> >::iterator it = begin_particle;
           it!=end_particle;)
        {
          try
            {
              const Point<dim> p_unit = this->get_mapping().transform_real_to_unit_cell(cell, it->second.get_location());
              if (GeometryInfo<dim>::is_inside_unit_cell(p_unit))
                {
                  it->second.set_reference_location(p_unit);
                  ++it;
                }
              else
                {
                  // The particle has left the cell
                  particles_out_of_cell.push_back(*it);

                  // Remove the lost particle and continue with next particle.
                  // Also make sure we do not invalidate the iterator we are increasing.
                  const typename std::multimap<types::LevelInd, Particle<dim> >::iterator particle_to_delete = it;
                  it++;
                  particles.erase(particle_to_delete);
                }
            }
          catch (typename Mapping<dim>::ExcTransformationFailed &)
            {
              // The particle has left the cell
              particles_out_of_cell.push_back(*it);

              // Remove the lost particle and continue with next particle.
              // Also make sure we do not invalidate the iterator we are increasing.
              const typename std::multimap<types::LevelInd, Particle<dim> >::iterator particle_to_delete = it;
              it++;
              particles.erase(particle_to_delete);
            }
        }
    }

    template <int dim>
    void
    World<dim>::setup_initial_state ()
    {
      // If we are in the first adaptive refinement cycle generate particles
      if (this->get_pre_refinement_step() == 0)
        generate_particles();

      // And initialize the tracer properties according to the initial
      // conditions on the current mesh
      initialize_particles();
    }

    template <int dim>
    void
    World<dim>::generate_particles()
    {
      TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Generate");

      generator->generate_particles(particles);
      update_n_global_particles();
      update_next_free_particle_index();
    }

    template <int dim>
    void
    World<dim>::initialize_particles()
    {
      // TODO: Change this loop over all cells to use the WorkStream interface
      if (property_manager->get_n_property_components() > 0)
        {
          TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Initialize properties");

          // Loop over all cells and initialize the particles cell-wise
          typename DoFHandler<dim>::active_cell_iterator
          cell = this->get_dof_handler().begin_active(),
          endc = this->get_dof_handler().end();

          for (; cell!=endc; ++cell)
            if (cell->is_locally_owned())
              {
                std::pair< const typename std::multimap<types::LevelInd,Particle <dim> >::iterator,
                    const typename std::multimap<types::LevelInd,Particle <dim> >::iterator>
                    particle_range_in_cell = particles.equal_range(std::make_pair(cell->level(),cell->index()));

                // Only initialize particles, if there are any in this cell
                if (particle_range_in_cell.first != particle_range_in_cell.second)
                  local_initialize_particles(particle_range_in_cell.first,
                                             particle_range_in_cell.second);
              }
          if (update_ghost_particles)
            exchange_ghost_particles();
        }
    }

    template <int dim>
    void
    World<dim>::update_particles()
    {
      // TODO: Change this loop over all cells to use the WorkStream interface

      if (property_manager->get_n_property_components() > 0)
        {
          TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Update properties");

          // Loop over all cells and update the particles cell-wise
          typename DoFHandler<dim>::active_cell_iterator
          cell = this->get_dof_handler().begin_active(),
          endc = this->get_dof_handler().end();

          for (; cell!=endc; ++cell)
            if (cell->is_locally_owned())
              {
                const std::pair< const typename std::multimap<types::LevelInd,Particle <dim> >::iterator,
                      const typename std::multimap<types::LevelInd,Particle <dim> >::iterator>
                      particle_range_in_cell = particles.equal_range(std::make_pair(cell->level(),cell->index()));

                // Only update particles, if there are any in this cell
                if (particle_range_in_cell.first != particle_range_in_cell.second)
                  local_update_particles(cell,
                                         particle_range_in_cell.first,
                                         particle_range_in_cell.second);
              }
        }
    }

    template <int dim>
    void
    World<dim>::advect_particles()
    {
      std::vector<std::pair<types::LevelInd,Particle <dim> > > particles_out_of_cell;
      particles_out_of_cell.reserve(particles.size());

      {
        // TODO: Change this loop over all cells to use the WorkStream interface
        TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Advect");

        // Loop over all cells and advect the particles cell-wise
        typename DoFHandler<dim>::active_cell_iterator
        cell = this->get_dof_handler().begin_active(),
        endc = this->get_dof_handler().end();

        for (; cell!=endc; ++cell)
          if (cell->is_locally_owned())
            {
              const std::pair< const typename std::multimap<types::LevelInd,Particle <dim> >::iterator,
                    const typename std::multimap<types::LevelInd,Particle <dim> >::iterator>
                    particle_range_in_cell = particles.equal_range(std::make_pair(cell->level(),cell->index()));

              // Only advect particles, if there are any in this cell
              if (particle_range_in_cell.first != particle_range_in_cell.second)
                local_advect_particles(cell,
                                       particle_range_in_cell.first,
                                       particle_range_in_cell.second,
                                       particles_out_of_cell);
            }
      }

      // Find the cells that the particles moved to
      sort_particles_in_subdomains_and_cells(particles_out_of_cell);
    }

    template <int dim>
    void
    World<dim>::advance_timestep()
    {
      do
        {
          advect_particles();
        }
      // Keep calling the integrator until it indicates it is finished
      while (integrator->new_integration_step());

      apply_particle_per_cell_bounds();

      // Update particle properties
      if (property_manager->need_update() == Property::update_time_step)
        update_particles();

      // Update the number of global particles if some have left the domain
      update_n_global_particles();

      // Now that all particle information was updated, exchange the new
      // ghost particles.
      if (update_ghost_particles)
        exchange_ghost_particles();
    }

    template <int dim>
    void
    World<dim>::save (std::ostringstream &os) const
    {
      aspect::oarchive oa (os);
      oa << (*this);
      output->save(os);
    }

    template <int dim>
    void
    World<dim>::load (std::istringstream &is)
    {
      aspect::iarchive ia (is);
      ia >> (*this);
      output->load(is);
    }

    template <int dim>
    void
    World<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Tracers");
        {
          prm.declare_entry ("Load balancing strategy", "repartition",
                             Patterns::MultipleSelection ("none|remove particles|add particles|"
                                                          "remove and add particles|repartition"),
                             "Strategy that is used to balance the computational"
                             "load across processors for adaptive meshes.");
          prm.declare_entry ("Minimum tracers per cell", "0",
                             Patterns::Integer (0),
                             "Lower limit for particle number per cell. This limit is "
                             "useful for adaptive meshes to prevent fine cells from being empty "
                             "of particles. It will be checked and enforced after mesh "
                             "refinement and after particle movement. "
                             "If there are "
                             "\\texttt{n\\_number\\_of\\_particles} $<$ \\texttt{min\\_particles\\_per\\_cell} "
                             "particles in one cell then "
                             "\\texttt{min\\_particles\\_per\\_cell} - \\texttt{n\\_number\\_of\\_particles} "
                             "particles are generated and randomly placed in "
                             "this cell. If the particles carry properties the "
                             "individual property plugins control how the "
                             "properties of the new particles are initialized.");
          prm.declare_entry ("Maximum tracers per cell", "100",
                             Patterns::Integer (0),
                             "Upper limit for particle number per cell. This limit is "
                             "useful for adaptive meshes to prevent coarse cells from slowing down "
                             "the whole model. It will be checked and enforced after mesh "
                             "refinement, after MPI transfer of particles and after particle "
                             "movement. If there are "
                             "\\texttt{n\\_number\\_of\\_particles} $>$ \\texttt{max\\_particles\\_per\\_cell} "
                             "particles in one cell then "
                             "\\texttt{n\\_number\\_of\\_particles} - \\texttt{max\\_particles\\_per\\_cell} "
                             "particles in this cell are randomly chosen and destroyed.");
          prm.declare_entry ("Tracer weight", "10",
                             Patterns::Integer (0),
                             "Weight that is associated with the computational load of "
                             "a single particle. The sum of particle weights will be added "
                             "to the sum of cell weights to determine the partitioning of "
                             "the mesh if the 'repartition' particle load balancing strategy "
                             "is selected. The optimal weight depends on the used "
                             "integrator and particle properties. In general for a more "
                             "expensive integrator and more expensive properties a larger "
                             "particle weight is recommended. Before adding the weights "
                             "of particles, each cell already carries a weight of 1000 to "
                             "account for the cost of field-based computations.");
          prm.declare_entry ("Update ghost particles", "false",
                             Patterns::Bool (),
                             "Some particle interpolation algorithms require knowledge "
                             "about particles in neighboring cells. To allow this, "
                             "particles in ghost cells need to be exchanged between the "
                             "processes neighboring this cell. This parameter determines "
                             "whether this transport is happening.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();

      Generator::declare_parameters<dim>(prm);
      Output::declare_parameters<dim>(prm);
      Integrator::declare_parameters<dim>(prm);
      Interpolator::declare_parameters<dim>(prm);
      Property::Manager<dim>::declare_parameters(prm);
    }


    template <int dim>
    void
    World<dim>::parse_parameters (ParameterHandler &prm)
    {
      // First do some error checking. The current algorithm does not find
      // the cells around particles, if the particles moved more than one
      // cell in one timestep and we are running in parallel, because they
      // skip the layer of ghost cells around our local domain. Assert this
      // is not possible.
      const double CFL_number = prm.get_double ("CFL number");
      const unsigned int n_processes = Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());

      AssertThrow((n_processes == 1) || (CFL_number <= 1.0),
                  ExcMessage("The current tracer algorithm does not work in "
                             "parallel if the CFL number is larger than 1.0, because "
                             "in this case tracers can move more than one cell's "
                             "diameter in one time step and therefore skip the layer "
                             "of ghost cells around the local subdomain."));

      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Tracers");
        {
          min_particles_per_cell = prm.get_integer("Minimum tracers per cell");
          max_particles_per_cell = prm.get_integer("Maximum tracers per cell");

          AssertThrow(min_particles_per_cell <= max_particles_per_cell,
                      ExcMessage("Please select a 'Minimum tracers per cell' parameter "
                                 "that is smaller than or equal to the 'Maximum tracers per cell' parameter."));

          tracer_weight = prm.get_integer("Tracer weight");

          update_ghost_particles = prm.get_bool("Update ghost particles");

          const std::vector<std::string> strategies = Utilities::split_string_list(prm.get ("Load balancing strategy"));
          particle_load_balancing = ParticleLoadBalancing::no_balancing;

          for (std::vector<std::string>::const_iterator strategy = strategies.begin(); strategy != strategies.end(); ++strategy)
            {
              if (*strategy == "remove particles")
                particle_load_balancing = typename ParticleLoadBalancing::Kind(particle_load_balancing | ParticleLoadBalancing::remove_particles);
              else if (*strategy == "add particles")
                particle_load_balancing = typename ParticleLoadBalancing::Kind(particle_load_balancing | ParticleLoadBalancing::add_particles);
              else if (*strategy == "remove and add particles")
                particle_load_balancing = typename ParticleLoadBalancing::Kind(particle_load_balancing | ParticleLoadBalancing::remove_and_add_particles);
              else if (*strategy == "repartition")
                particle_load_balancing = typename ParticleLoadBalancing::Kind(particle_load_balancing | ParticleLoadBalancing::repartition);
              else if (*strategy == "none")
                {
                  particle_load_balancing = ParticleLoadBalancing::no_balancing;
                  AssertThrow(strategies.size() == 1,
                              ExcMessage("The particle load balancing strategy 'none' is not compatible "
                                         "with any other strategy, yet it seems another is selected as well. "
                                         "Please check the parameter file."));
                }
              else
                AssertThrow(false,
                            ExcMessage("The 'Load balancing strategy' parameter contains an unknown value: <" + *strategy
                                       + ">. This value does not correspond to any known load balancing strategy. Possible values "
                                       "are listed in the corresponding manual subsection."));
            }

        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();

      TimerOutput::Scope timer_section(this->get_computing_timer(), "Particles: Initialization");

      // Create a generator object depending on what the parameters specify
      generator.reset(Generator::create_particle_generator<dim> (prm));
      if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(generator.get()))
        sim->initialize_simulator (this->get_simulator());
      generator->parse_parameters(prm);

      // Create an output object depending on what the parameters specify
      output.reset(Output::create_particle_output<dim>
                   (prm));

      // We allow to not generate any output plugin, in which case output is
      // a null pointer. Only initialize output if it was created.
      if (output)
        {
          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(output.get()))
            sim->initialize_simulator (this->get_simulator());
          output->parse_parameters(prm);
          output->initialize();
        }

      // Create an integrator object depending on the specified parameter
      integrator.reset(Integrator::create_particle_integrator<dim> (prm));
      if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(integrator.get()))
        sim->initialize_simulator (this->get_simulator());
      integrator->parse_parameters(prm);

      // Create an interpolator object depending on the specified parameter
      interpolator.reset(Interpolator::create_particle_interpolator<dim> (prm));
      if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(interpolator.get()))
        sim->initialize_simulator (this->get_simulator());
      interpolator->parse_parameters(prm);

      // Creaty an property_manager object and initialize its properties
      property_manager.reset(new Property::Manager<dim> ());
      SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(property_manager.get());
      sim->initialize_simulator (this->get_simulator());
      property_manager->parse_parameters(prm);
      property_manager->initialize();
    }
  }
}


// explicit instantiation of the functions we implement in this file
namespace aspect
{
  namespace Particle
  {
#define INSTANTIATE(dim) \
  template class World<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)
  }
}
