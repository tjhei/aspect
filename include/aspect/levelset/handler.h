/*
  Copyright (C) 2020 - 2021 by the authors of the ASPECT code.

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

#ifndef _aspect_levelset_handler_h
#define _aspect_levelset_handler_h

#include <aspect/simulator_access.h>
#include <aspect/simulator.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/time_stepping.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
using namespace dealii;

namespace aspect
{

  namespace internal {
    using Number = double;
    using vectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

    template <int dim, int degree, int velocity_degree, int n_points_1d>
    class LevelsetOperator
    {
    public:
      LevelsetOperator(TimerOutput &timer_output);

      void
      reinit(const Mapping<dim> &                                 mapping,
             const std::vector<const DoFHandler<dim> *>          &dof_handlers,
             const std::vector<const AffineConstraints<double> *> constraints,
             const std::vector<Quadrature<1>>                     quadratures);

      void initialize_vector(vectorType &vector,
                             const unsigned int no_dof = 0) const;

      void apply_forward_euler(
        const double time_step,
        const std::vector<vectorType *> &src,
        vectorType &dst) const;

      vectorType
      apply(const double current_time,
            const double step_time,
            const double old_time_step,
            const vectorType &old_velocity,
            const vectorType &old_old_velocity,
            const vectorType &src) const;

      void perform_stage(
        const double                                     time_step,
        const double                                     bi,
        const double                                     ai,
        const vectorType levelset_old_solution,
        const std::vector<vectorType *> &src,
        vectorType &dst) const;

    private:
      MatrixFree<dim, Number> data;

      TimerOutput &timer;

      VectorizedArray<Number> lambda;

      void local_apply_inverse_mass_matrix(
        const MatrixFree<dim, Number> &                   data,
        vectorType &      dst,
        const vectorType &src,
        const std::pair<unsigned int, unsigned int> &     cell_range) const;

      void local_apply_cell(
        const MatrixFree<dim, Number> &                                  data,
        vectorType &                     dst,
        const std::vector<vectorType *> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

      void local_apply_face(
        const MatrixFree<dim, Number> &                                  data,
        vectorType &                     dst,
        const std::vector<vectorType *> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

      void local_apply_boundary_face(
        const MatrixFree<dim, Number> &                                  data,
        vectorType &                     dst,
        const std::vector<vectorType *> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;
    };

  }

  template <int dim> class Simulator;

  /**
   * A member class that isolates the functions and variables that deal
   * with the Volume of Fluid implementation. If Volume of Fluid interface
   * tracking is not active, there is no instantiation of this class at
   * all.
   */
  template <int dim>
  class LevelsetHandler : public SimulatorAccess<dim>
  {
    public:
      /**
       * Standard initial constructor
       */
      LevelsetHandler(Simulator<dim> &simulator, ParameterHandler &prm);

      /**
       * Declare the parameters this class takes through input files.
       */
      static
      void declare_parameters (ParameterHandler &prm);

      /**
       * Read the parameters this class declares from the parameter file.
       */
      void parse_parameters (ParameterHandler &prm);

      /**
       * Do necessary internal initialization that is dependent on having the
       * simulator and Finite Element initialized.
       */
      void initialize (ParameterHandler &prm);

      /**
       * Perform the advection step for the specified field.
       */
      void solve(const typename Simulator<dim>::AdvectionField &advection_field);

    private:
      /**
       * Parent simulator
       */
      Simulator<dim> &sim;

      DoFHandler<dim> dof_handler_v;
      DoFHandler<dim> dof_handler_levelset;

      FESystem<dim> fe_v;
      FE_DGQ<dim> fe_levelset;


      friend class Simulator<dim>;
  };

}

#endif
