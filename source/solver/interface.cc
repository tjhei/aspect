/*
  Copyright (C) 2011 - 2016 by the authors of the ASPECT code.

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


#include <aspect/solver/interface.h>

#include <typeinfo>


namespace aspect
{
  namespace Solver
  {
// ------------------------------ Interface -----------------------------

    template <int dim>
    Interface<dim>::~Interface ()
    {}

    template <int dim>
    void
    Interface<dim>::initialize ()
    {}


    template <int dim>
    void
    Interface<dim>::update ()
    {}


    template <int dim>
    void
    Interface<dim>::execute (SolveInfo<dim> &/*info*/) const
    {
    }



    template <int dim>
    void
    Interface<dim>::declare_parameters (ParameterHandler &)
    {}



    template <int dim>
    void
    Interface<dim>::parse_parameters (ParameterHandler &)
    {}



// ------------------------------ Manager -----------------------------

    template <int dim>
    Manager<dim>::~Manager()
    {}



    template <int dim>
    void
    Manager<dim>::update ()
    {
      Assert (solver_objects.size() > 0, ExcInternalError());

      // call the update() functions of all
      // refinement plugins.
      unsigned int index = 0;
      for (typename std::list<std_cxx11::shared_ptr<Interface<dim> > >::const_iterator
           p = solver_objects.begin();
           p != solver_objects.end(); ++p, ++index)
        {
          try
            {
              (*p)->update ();
            }

          // plugins that throw exceptions usually do not result in
          // anything good because they result in an unwinding of the stack
          // and, if only one processor triggers an exception, the
          // destruction of objects often causes a deadlock. thus, if
          // an exception is generated, catch it, print an error message,
          // and abort the program
          catch (std::exception &exc)
            {
              std::cerr << std::endl << std::endl
                        << "----------------------------------------------------"
                        << std::endl;
              std::cerr << "Exception on MPI process <"
                        << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                        << "> while running mesh refinement plugin <"
                        << typeid(**p).name()
                        << ">: " << std::endl
                        << exc.what() << std::endl
                        << "Aborting!" << std::endl
                        << "----------------------------------------------------"
                        << std::endl;

              // terminate the program!
              MPI_Abort (MPI_COMM_WORLD, 1);
            }
          catch (...)
            {
              std::cerr << std::endl << std::endl
                        << "----------------------------------------------------"
                        << std::endl;
              std::cerr << "Exception on MPI process <"
                        << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                        << "> while running mesh refinement plugin <"
                        << typeid(**p).name()
                        << ">: " << std::endl;
              std::cerr << "Unknown exception!" << std::endl
                        << "Aborting!" << std::endl
                        << "----------------------------------------------------"
                        << std::endl;

              // terminate the program!
              MPI_Abort (MPI_COMM_WORLD, 1);
            }
        }
    }



    template <int dim>
    void
    Manager<dim>::execute (SolveInfo<dim> &info) const
    {
      Assert (solver_objects.size() > 0, ExcInternalError());

      unsigned int index = 0;
      for (typename std::list<std_cxx11::shared_ptr<Interface<dim> > >::const_iterator
           p = solver_objects.begin();
           p != solver_objects.end(); ++p, ++index)
        {
          std::cout << "executing solver " << index << " " << typeid(**p).name() << std::endl;
          try
            {
              (*p)->execute (info);
            std::cout << "finished solver " << index << " " << typeid(**p).name() << std::endl;
            }
          // plugins that throw exceptions usually do not result in
          // anything good because they result in an unwinding of the stack
          // and, if only one processor triggers an exception, the
          // destruction of objects often causes a deadlock. thus, if
          // an exception is generated, catch it, print an error message,
          // and abort the program
          catch (std::exception &exc)
            {
              std::cerr << std::endl << std::endl
                        << "----------------------------------------------------"
                        << std::endl;
              std::cerr << "Exception on MPI process <"
                        << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                        << "> while running solver plugin <"
                        << typeid(**p).name()
                        << ">: " << std::endl
                        << exc.what() << std::endl
                        << "Aborting!" << std::endl
                        << "----------------------------------------------------"
                        << std::endl;

              // terminate the program!
              MPI_Abort (MPI_COMM_WORLD, 1);
            }
          catch (...)
            {
              std::cerr << std::endl << std::endl
                        << "----------------------------------------------------"
                        << std::endl;
              std::cerr << "Exception on MPI process <"
                        << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                        << "> while running solver plugin <"
                        << typeid(**p).name()
                        << ">: " << std::endl;
              std::cerr << "Unknown exception!" << std::endl
                        << "Aborting!" << std::endl
                        << "----------------------------------------------------"
                        << std::endl;

              // terminate the program!
              MPI_Abort (MPI_COMM_WORLD, 1);
            }
        }

      /*

    switch (parameters.nonlinear_solver)
      {
        case NonlinearSolver::IMPES:
        {
          //We do the free surface execution at the beginning of the timestep for a specific reason.
          //The time step size is calculated AFTER the whole solve_timestep() function.  If we call
          //free_surface_execute() after the Stokes solve, it will be before we know what the appropriate
          //time step to take is, and we will timestep the boundary incorrectly.
          if (parameters.free_surface_enabled)
            free_surface->execute ();

          assemble_advection_system (AdvectionField::temperature());
          solve_advection(AdvectionField::temperature());

          if (parameters.use_discontinuous_temperature_discretization
              && parameters.use_limiter_for_discontinuous_temperature_solution)
            apply_limiter_to_dg_solutions(AdvectionField::temperature());

          current_linearization_point.block(introspection.block_indices.temperature)
            = solution.block(introspection.block_indices.temperature);

          for (unsigned int c=0; c < parameters.n_compositional_fields; ++c)
            {
              assemble_advection_system (AdvectionField::composition(c));
              solve_advection(AdvectionField::composition(c));
              if (parameters.use_discontinuous_composition_discretization
                  && parameters.use_limiter_for_discontinuous_composition_solution)
                apply_limiter_to_dg_solutions(AdvectionField::composition(c));
            }

          for (unsigned int c=0; c<parameters.n_compositional_fields; ++c)
            current_linearization_point.block(introspection.block_indices.compositional_fields[c])
              = solution.block(introspection.block_indices.compositional_fields[c]);

          // the Stokes matrix depends on the viscosity. if the viscosity
          // depends on other solution variables, then after we need to
          // update the Stokes matrix in every time step and so need to set
          // the following flag. if we change the Stokes matrix we also
          // need to update the Stokes preconditioner.
          if (stokes_matrix_depends_on_solution() == true)
            rebuild_stokes_matrix = rebuild_stokes_preconditioner = true;

          assemble_stokes_system();
          build_stokes_preconditioner();
          solve_stokes();

          break;
        }
        case NonlinearSolver::Stokes_only:
        {
          unsigned int iteration = 0;

          do
            {
              // the Stokes matrix depends on the viscosity. if the viscosity
              // depends on other solution variables, then we need to
              // update the Stokes matrix in every iteration and so need to set
              // the rebuild_stokes_matrix flag. if we change the Stokes matrix we also
              // need to update the Stokes preconditioner.
              //
              // there is a similar case where this nonlinear solver can be used, namely for
              // compressible models. in that case, the matrix does not depend on
              // the previous solution, but we still need to iterate since the right
              // hand side depends on it. in those cases, the matrix does not change,
              // but if we have to repeat computing the right hand side, we need to
              // also rebuild the matrix if we end up with inhomogenous velocity
              // boundary conditions (i.e., if there are prescribed velocity boundary
              // indicators)
              if ((stokes_matrix_depends_on_solution() == true)
                  ||
                  (parameters.prescribed_velocity_boundary_indicators.size() > 0))
                rebuild_stokes_matrix = true;
              if (stokes_matrix_depends_on_solution() == true)
                rebuild_stokes_preconditioner = true;

              assemble_stokes_system();
              build_stokes_preconditioner();
              const double stokes_residual = solve_stokes();
              current_linearization_point = solution;

              pcout << "      Nonlinear Stokes residual: " << stokes_residual << std::endl;
              if (stokes_residual < 1e-8)
                break;

              ++iteration;
            }
          while (! ((iteration >= parameters.max_nonlinear_iterations) // regular timestep
                    ||
                    ((pre_refinement_step < parameters.initial_adaptive_refinement) // pre-refinement
                     &&
                     (iteration >= parameters.max_nonlinear_iterations_in_prerefinement))));
          break;
        }


        case NonlinearSolver::iterated_IMPES:
        {
          double initial_temperature_residual = 0;
          double initial_stokes_residual      = 0;
          std::vector<double> initial_composition_residual (parameters.n_compositional_fields,0);

          unsigned int iteration = 0;

          do
            {
              assemble_advection_system(AdvectionField::temperature());

              if (iteration == 0)
                initial_temperature_residual = system_rhs.block(introspection.block_indices.temperature).l2_norm();

              const double temperature_residual = solve_advection(AdvectionField::temperature());

              current_linearization_point.block(introspection.block_indices.temperature)
                = solution.block(introspection.block_indices.temperature);
              rebuild_stokes_matrix = true;
              std::vector<double> composition_residual (parameters.n_compositional_fields,0);

              for (unsigned int c=0; c<parameters.n_compositional_fields; ++c)
                {
                  assemble_advection_system (AdvectionField::composition(c));

                  if (iteration == 0)
                    initial_composition_residual[c] = system_rhs.block(introspection.block_indices.compositional_fields[c]).l2_norm();

                  composition_residual[c]
                    = solve_advection(AdvectionField::composition(c));
                }

              // for consistency we update the current linearization point only after we have solved
              // all fields, so that we use the same point in time for every field when solving
              for (unsigned int c=0; c<parameters.n_compositional_fields; ++c)
                current_linearization_point.block(introspection.block_indices.compositional_fields[c])
                  = solution.block(introspection.block_indices.compositional_fields[c]);

              // the Stokes matrix depends on the viscosity. if the viscosity
              // depends on other solution variables, then after we need to
              // update the Stokes matrix in every time step and so need to set
              // the following flag. if we change the Stokes matrix we also
              // need to update the Stokes preconditioner.
              if (stokes_matrix_depends_on_solution() == true)
                rebuild_stokes_matrix = rebuild_stokes_preconditioner = true;

              assemble_stokes_system();
              build_stokes_preconditioner();

              if (iteration == 0)
                initial_stokes_residual = compute_initial_stokes_residual();

              const double stokes_residual = solve_stokes();

              current_linearization_point = solution;

              // write the residual output in the same order as the output when
              // solving the equations
              pcout << "      Nonlinear residuals: " << temperature_residual;

              for (unsigned int c=0; c<parameters.n_compositional_fields; ++c)
                pcout << ", " << composition_residual[c];

              pcout << ", " << stokes_residual;

              pcout << std::endl;

              double max = 0.0;
              for (unsigned int c=0; c<parameters.n_compositional_fields; ++c)
                {
                  // in models with melt migration the melt advection equation includes the divergence of the velocity
                  // and can not be expected to converge to a smaller value than the residual of the Stokes equation.
                  // thus, we set a threshold for the initial composition residual.
                  // this only plays a role if the right-hand side of the advection equation is very small.
                  const double threshold = (parameters.include_melt_transport && c == introspection.compositional_index_for_name("porosity")
                                            ?
                                            parameters.linear_stokes_solver_tolerance * time_step
                                            :
                                            0.0);
                  if (initial_composition_residual[c]>threshold)
                    max = std::max(composition_residual[c]/initial_composition_residual[c],max);
                }

              if (initial_stokes_residual>0)
                max = std::max(stokes_residual/initial_stokes_residual, max);
              if (initial_temperature_residual>0)
                max = std::max(temperature_residual/initial_temperature_residual, max);
              pcout << "      Total relative nonlinear residual: " << max << std::endl;
              pcout << std::endl
                    << std::endl;
              if (max < parameters.nonlinear_tolerance)
                break;

              ++iteration;
//TODO: terminate here if the number of iterations is too large and we see no convergence
            }
          while (! ((iteration >= parameters.max_nonlinear_iterations) // regular timestep
                    ||
                    ((pre_refinement_step < parameters.initial_adaptive_refinement) // pre-refinement
                     &&
                     (iteration >= parameters.max_nonlinear_iterations_in_prerefinement))));

          break;
        }

        case NonlinearSolver::iterated_Stokes:
        {
          if (parameters.free_surface_enabled)
            free_surface->execute ();

          // solve the temperature and composition systems once...
          assemble_advection_system (AdvectionField::temperature());
          solve_advection(AdvectionField::temperature());
          current_linearization_point.block(introspection.block_indices.temperature)
            = solution.block(introspection.block_indices.temperature);

          for (unsigned int c=0; c<parameters.n_compositional_fields; ++c)
            {
              assemble_advection_system (AdvectionField::composition(c));
              solve_advection(AdvectionField::composition(c));
            }

          for (unsigned int c=0; c<parameters.n_compositional_fields; ++c)
            current_linearization_point.block(introspection.block_indices.compositional_fields[c])
              = solution.block(introspection.block_indices.compositional_fields[c]);

          // residual vector (only for the velocity)
          LinearAlgebra::Vector residual (introspection.index_sets.system_partitioning[0], mpi_communicator);
          LinearAlgebra::Vector tmp (introspection.index_sets.system_partitioning[0], mpi_communicator);

          // ...and then iterate the solution of the Stokes system
          double initial_stokes_residual = 0;
          for (unsigned int i=0; (! ((i >= parameters.max_nonlinear_iterations) // regular timestep
                                     ||
                                     ((pre_refinement_step < parameters.initial_adaptive_refinement) // pre-refinement
                                      &&
                                      (i >= parameters.max_nonlinear_iterations_in_prerefinement)))); ++i)
            {
              // rebuild the matrix if it actually depends on the solution
              // of the previous iteration.
              if ((stokes_matrix_depends_on_solution() == true)
                  ||
                  (parameters.prescribed_velocity_boundary_indicators.size() > 0))
                rebuild_stokes_matrix = rebuild_stokes_preconditioner = true;

              assemble_stokes_system();
              build_stokes_preconditioner();

              if (i==0)
                initial_stokes_residual = compute_initial_stokes_residual();

              const double stokes_residual = solve_stokes();

              pcout << "   Residual after nonlinear iteration " << i+1 << ": " << stokes_residual/initial_stokes_residual << std::endl;
              if (stokes_residual/initial_stokes_residual < parameters.nonlinear_tolerance)
                {
                  break; // convergence reached, exit nonlinear iterations.
                }

              current_linearization_point.block(introspection.block_indices.velocities)
                = solution.block(introspection.block_indices.velocities);
              if (introspection.block_indices.velocities != introspection.block_indices.pressure)
                current_linearization_point.block(introspection.block_indices.pressure)
                  = solution.block(introspection.block_indices.pressure);

              pcout << std::endl;
            }

          break;
        }

        case NonlinearSolver::Advection_only:
        {
          // Identical to IMPES except does not solve Stokes equation
          if (parameters.free_surface_enabled)
            free_surface->execute ();

          LinearAlgebra::BlockVector distributed_stokes_solution (introspection.index_sets.system_partitioning, mpi_communicator);

          VectorFunctionFromVectorFunctionObject<dim> func(std_cxx1x::bind (&PrescribedStokesSolution::Interface<dim>::stokes_solution,
                                                                            std_cxx1x::cref(*prescribed_stokes_solution),
                                                                            std_cxx1x::_1,
                                                                            std_cxx1x::_2),
                                                           0,
                                                           dim+1, //velocity and pressure
                                                           introspection.n_components);

          VectorTools::interpolate (mapping, dof_handler, func, distributed_stokes_solution);

          const unsigned int block_vel = introspection.block_indices.velocities;
          const unsigned int block_p = introspection.block_indices.pressure;

          // distribute hanging node and
          // other constraints
          current_constraints.distribute (distributed_stokes_solution);

          solution.block(block_vel) = distributed_stokes_solution.block(block_vel);
          solution.block(block_p) = distributed_stokes_solution.block(block_p);

          assemble_advection_system (AdvectionField::temperature());
          solve_advection(AdvectionField::temperature());

          current_linearization_point.block(introspection.block_indices.temperature)
            = solution.block(introspection.block_indices.temperature);

          for (unsigned int c=0; c<parameters.n_compositional_fields; ++c)
            {
              assemble_advection_system (AdvectionField::composition(c));
              solve_advection(AdvectionField::composition(c));
              current_linearization_point.block(introspection.block_indices.compositional_fields[c])
                = solution.block(introspection.block_indices.compositional_fields[c]);
            }

          break;
        }

        default:
          Assert (false, ExcNotImplemented());
      }*/
    }







// -------------------------------- Deal with registering plugins and automating
// -------------------------------- their setup and selection at run time

    namespace
    {
      std_cxx11::tuple
      <void *,
      void *,
      internal::Plugins::PluginList<Interface<2> >,
      internal::Plugins::PluginList<Interface<3> > > registered_plugins;
    }



    template <int dim>
    void
    Manager<dim>::declare_parameters (ParameterHandler &prm)
    {


      prm.enter_subsection("Solver");
      {

        // construct a string for Patterns::MultipleSelection that
        // contains the names of all registered plugins
        const std::string pattern_of_names
          = std_cxx11::get<dim>(registered_plugins).get_pattern_of_names ();

        prm.declare_entry("Strategy",
                          "extrapolate, temperature, stokes, compositions",
                          Patterns::MultipleSelection(pattern_of_names),
                          "A comma separated list of ...\n\n"
                          "The following criteria are available:\n\n"
                          +
                          std_cxx11::get<dim>(registered_plugins).get_description_string());

      }
      prm.leave_subsection();

      // now declare the parameters of each of the registered
      // plugins in turn
      std_cxx11::get<dim>(registered_plugins).declare_parameters (prm);
    }



    template <int dim>
    void
    Manager<dim>::parse_parameters (ParameterHandler &prm)
    {
      Assert (std_cxx11::get<dim>(registered_plugins).plugins != 0,
              ExcMessage ("No mesh refinement plugins registered!?"));

      // find out which plugins are requested and the various other
      // parameters we declare here
      std::vector<std::string> plugin_names;
      prm.enter_subsection("Solver");
      {
        plugin_names
          = Utilities::split_string_list(prm.get("Strategy"));

      }
      prm.leave_subsection();

      // go through the list, create objects and let them parse
      // their own parameters
      AssertThrow (plugin_names.size() >= 1,
                   ExcMessage ("You need to provide at least one mesh refinement criterion in the input file!"));
      for (unsigned int name=0; name<plugin_names.size(); ++name)
        {
          solver_objects.push_back (std_cxx11::shared_ptr<Interface<dim> >
                                             (std_cxx11::get<dim>(registered_plugins)
                                              .create_plugin (plugin_names[name],
                                                              "Mesh refinement::Refinement criteria merge operation")));

          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(&*solver_objects.back()))
            sim->initialize_simulator (this->get_simulator());

          solver_objects.back()->parse_parameters (prm);
          solver_objects.back()->initialize ();
        }
    }


    template <int dim>
    void
    Manager<dim>::register_solver (const std::string &name,
                                                      const std::string &description,
                                                      void (*declare_parameters_function) (ParameterHandler &),
                                                      Interface<dim> *(*factory_function) ())
    {
      std_cxx11::get<dim>(registered_plugins).register_plugin (name,
                                                               description,
                                                               declare_parameters_function,
                                                               factory_function);
    }

  }
}


// explicit instantiations
namespace aspect
{
  namespace internal
  {
    namespace Plugins
    {
      template <>
      std::list<internal::Plugins::PluginList<Solver::Interface<2> >::PluginInfo> *
      internal::Plugins::PluginList<Solver::Interface<2> >::plugins = 0;
      template <>
      std::list<internal::Plugins::PluginList<Solver::Interface<3> >::PluginInfo> *
      internal::Plugins::PluginList<Solver::Interface<3> >::plugins = 0;
    }
  }

  namespace Solver
  {
#define INSTANTIATE(dim) \
  template class Interface<dim>; \
  template class Manager<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)
  }
}
