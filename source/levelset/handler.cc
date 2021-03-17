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

#include <aspect/global.h>
#include <aspect/levelset/handler.h>


#include <deal.II/dofs/dof_renumbering.h>

using namespace dealii;

namespace aspect
{
  namespace internal
  {

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE Tensor<1, dim, Number>
                                 levelset_flux(const Number &phi, const Tensor<1, dim, Number> &velocity)
    {
      Tensor<1, dim, Number> flux(velocity);
      flux *= phi;

      return flux;
    }


    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      levelset_numerical_flux(const Number &                u_m,
                              const Number &                u_p,
                              const Tensor<1, dim, Number> &normal,
                              const Tensor<1, dim, Number> &velocity,
                              const Number &                lambda)
    {
      return 0.5 * (u_m + u_p) * (velocity * normal) + 0.5 * lambda * (u_m - u_p);
    }

    template <int dim, typename Number>
    VectorizedArray<Number>
    evaluate_velocity(const Function<dim> &                      function,
                      const Point<dim, VectorizedArray<Number>> &p_vectorized,
                      const unsigned int                         component)
    {
      VectorizedArray<Number> result;
      for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
        {
          Point<dim> p;
          for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vectorized[d][v];
          result[v] = function.value(p, component);
        }
      return result;
    }


    template <int dim, typename Number, int n_components = dim>
    Tensor<1, n_components, VectorizedArray<Number>>
    evaluate_velocity(const Function<dim> &                      function,
                      const Point<dim, VectorizedArray<Number>> &p_vectorized)
    {
      AssertDimension(function.n_components, n_components);
      Tensor<1, n_components, VectorizedArray<Number>> result;
      for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
        {
          Point<dim> p;
          for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vectorized[d][v];
          for (unsigned int d = 0; d < n_components; ++d)
            result[d][v] = function.value(p, d);
        }
      return result;
    }


    template <int dim, int degree, int velocity_degree, int n_points_1d>
    LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::LevelsetOperator(
      TimerOutput &timer)
      : timer(timer)
    {}


    template <int dim, int degree, int velocity_degree, int n_points_1d>
    void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::reinit(
      const Mapping<dim> &                                 mapping,
      const std::vector<const DoFHandler<dim> *>          &dof_handlers,
      const std::vector<const AffineConstraints<double> *> constraints,
      const std::vector<Quadrature<1>>                     quadratures)
    {
      typename dealii::MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points |
         update_values);
      additional_data.mapping_update_flags_inner_faces =
        (update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
      additional_data.mapping_update_flags_boundary_faces =
        (update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
      additional_data.tasks_parallel_scheme =
        dealii::MatrixFree<dim, Number>::AdditionalData::none;

      data.reinit(
        mapping, dof_handlers, constraints, quadratures, additional_data);
    }

    template <int dim, int degree, int velocity_degree, int n_points_1d>
    void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
      initialize_vector(vectorType &vector,
                        const unsigned int                          no_dof) const
    {
      data.initialize_dof_vector(vector, no_dof);
    }


    template <int dim, int degree, int velocity_degree, int n_points_1d>
    void
    LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::local_apply_cell(
      const dealii::MatrixFree<dim, Number> &                                  data,
      vectorType &                     dst,
      const std::vector<vectorType *> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, degree, n_points_1d, 1, Number>            phi(data, 0);
      FEEvaluation<dim, velocity_degree, n_points_1d, dim, Number> vel_phi(data,
                                                                           1,
                                                                           0,
                                                                           0);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(*src[0], dealii::EvaluationFlags::values);

          vel_phi.reinit(cell);
          vel_phi.gather_evaluate(*src[1], dealii::EvaluationFlags::values);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto w_q   = phi.get_value(q);
              const auto vel_q = vel_phi.get_value(q);
              phi.submit_gradient(levelset_flux(w_q, vel_q), q);
            }
          phi.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
        }
    }


    template <int dim, int degree, int velocity_degree, int n_points_1d>
    void
    LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::local_apply_face(
      const dealii::MatrixFree<dim, Number> &,
      vectorType &                     dst,
      const std::vector<vectorType *> &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, degree, n_points_1d, 1, Number> phi_m(data, true, 0);
      FEFaceEvaluation<dim, degree, n_points_1d, 1, Number> phi_p(data, false, 0);
      FEFaceEvaluation<dim, velocity_degree, n_points_1d, dim, Number> vel_phi(
        data, true, 1, 0, 0);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          phi_p.reinit(face);
          phi_p.gather_evaluate(*src[0], dealii::EvaluationFlags::values);

          phi_m.reinit(face);
          phi_m.gather_evaluate(*src[0], dealii::EvaluationFlags::values);

          vel_phi.reinit(face);
          vel_phi.gather_evaluate(*src[1], dealii::EvaluationFlags::values);
          // compute local lambda
          VectorizedArray<Number> lambda = 0.;
          for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
            {
              const auto velq = vel_phi.get_value(q);
              lambda          = std::max(lambda, velq.norm());
            }

          for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
            {
              const auto numerical_flux =
                levelset_numerical_flux<dim>(phi_m.get_value(q),
                                             phi_p.get_value(q),
                                             phi_m.get_normal_vector(q),
                                             vel_phi.get_value(q),
                                             lambda);
              phi_m.submit_value(-numerical_flux, q);
              phi_p.submit_value(numerical_flux, q);
            }

          phi_p.integrate_scatter(dealii::EvaluationFlags::values, dst);
          phi_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
        }
    }

    template <int dim, int degree, int velocity_degree, int n_points_1d>
    void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
      local_apply_boundary_face(
        const dealii::MatrixFree<dim, Number> &,
        vectorType &                     dst,
        const std::vector<vectorType *> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, degree, n_points_1d, 1, Number> phi(data, true, 0);
      FEFaceEvaluation<dim, velocity_degree, n_points_1d, dim, Number> vel_phi(
        data, true, 1, 0, 0);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          phi.reinit(face);
          phi.gather_evaluate(*src[0], dealii::EvaluationFlags::values);

          vel_phi.reinit(face);
          vel_phi.gather_evaluate(*src[1], dealii::EvaluationFlags::values);

          // compute local lambda
          VectorizedArray<Number> lambda = 0.;
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto velq = vel_phi.get_value(q);
              lambda          = std::max(lambda, velq.norm());
            }

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto w_m    = phi.get_value(q);
              const auto normal = phi.get_normal_vector(q);
              const auto vel_q  = vel_phi.get_value(q);
              auto       flux =
                levelset_numerical_flux<dim>(w_m, w_m, normal, vel_q, lambda);
              phi.submit_value(-flux, q);
            }

          phi.integrate_scatter(dealii::EvaluationFlags::values, dst);
        }
    }

    template <int dim, int degree, int velocity_degree, int n_points_1d>
    void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
      local_apply_inverse_mass_matrix(
        const dealii::MatrixFree<dim, Number> &,
        vectorType &      dst,
        const vectorType &src,
        const std::pair<unsigned int, unsigned int> &     cell_range) const
    {
      FEEvaluation<dim, degree, degree + 1, 1, Number> phi(data, 0, 1);
      dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, 1, Number>
        inverse(phi);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);

          inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

          phi.set_dof_values(dst);
        }
    }


    template <int dim, int degree, int velocity_degree, int n_points_1d>
    void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
      apply_forward_euler(
        const double time_step,
        const std::vector<vectorType *> &src,
        vectorType &dst) const
    {
      {
        TimerOutput::Scope t(timer, "apply - integrals");

        data.loop(&LevelsetOperator::local_apply_cell,
                  &LevelsetOperator::local_apply_face,
                  &LevelsetOperator::local_apply_boundary_face,
                  this,
                  dst,
                  src,
                  true,
                  dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                  dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
      }

      {
        TimerOutput::Scope t(timer, "apply - inverse mass");

        data.cell_loop(
          &LevelsetOperator::local_apply_inverse_mass_matrix,
          this,
          dst,
          dst,
          std::function<void(const unsigned int, const unsigned int)>(),
          [&](const unsigned int start_range, const unsigned int end_range) {
            const Number ts = time_step;
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number old_sol_i = src[0]->local_element(i);
                const Number sol_i     = dst.local_element(i);
                dst.local_element(i)   = old_sol_i + ts * sol_i;
              }
          });
      }
    }

    template <int dim, int degree, int velocity_degree, int n_points_1d>
    void
    LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::perform_stage(
      const double                                     time_step,
      const double                                     bi,
      const double                                     ai,
      const vectorType levelset_old_solution,
      const std::vector<vectorType *>
        &                                         src /*ui and velocity*/,
      vectorType &dst /*next ui*/) const
    {
      {
        TimerOutput::Scope t(timer, "apply - integrals");

        data.loop(&LevelsetOperator::local_apply_cell,
                  &LevelsetOperator::local_apply_face,
                  &LevelsetOperator::local_apply_boundary_face,
                  this,
                  dst,
                  src,
                  true,
                  dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                  dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
      }

      {
        TimerOutput::Scope t(timer, "apply - inverse mass");

        data.cell_loop(
          &LevelsetOperator::local_apply_inverse_mass_matrix,
          this,
          dst,
          dst,
          std::function<void(const unsigned int, const unsigned int)>(),
          [&](const unsigned int start_range, const unsigned int end_range) {
            const Number ts = time_step;
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number u_i       = dst.local_element(i);
                const Number old_u_i   = src[0]->local_element(i);
                const Number old_sol_i = levelset_old_solution.local_element(i);
                dst.local_element(i) = ai * old_sol_i + bi * (old_u_i + ts * u_i);
              }
          });
      }
    }

    template <int dim, int degree, int velocity_degree, int n_points_1d>
    vectorType
    LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
        apply(
            const double stage_time,
            const double step_time,
            const double old_time_step,
            const vectorType &old_velocity,
            const vectorType &old_old_velocity,
            const vectorType &src) const
    {
      TimerOutput::Scope t(timer, "apply function");
      vectorType dst(src);
      vectorType tmp_src(src);
      tmp_src = src;
      vectorType advect_velocity(old_velocity);
      advect_velocity = old_velocity;
      {
        //extrapolate velocity if stage_time > step_time
        TimerOutput::Scope t(timer, "extrapolate velocity");
        if (stage_time - step_time > 1e-12)
      {
        const double time_step = stage_time - step_time;
        const double time_step_factor = time_step / old_time_step;
        advect_velocity *= (1. + time_step_factor);
        // advect_velocity.equ(1. + time_step_factor, old_velocity);
        advect_velocity.add(-time_step_factor, old_old_velocity);
      }
      }
      const std::vector<vectorType *> src_and_vel(
                                          {&tmp_src, &advect_velocity});

      {
        TimerOutput::Scope t(timer, "apply - integrals");

        data.loop(&LevelsetOperator::local_apply_cell,
                  &LevelsetOperator::local_apply_face,
                  &LevelsetOperator::local_apply_boundary_face,
                  this,
                  dst,
                  src_and_vel,
                  true,
                  dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                  dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
      }

      {
        TimerOutput::Scope t(timer, "apply - inverse mass");

        data.cell_loop(
          &LevelsetOperator::local_apply_inverse_mass_matrix,
          this,
          dst,
          dst
          );
      }
      return dst;
    }

  } // name space internal



  template <int dim>
  LevelsetHandler<dim>::LevelsetHandler (Simulator<dim> &simulator,
                                         ParameterHandler &prm)
    : sim (simulator),

      fe_v (FE_Q<dim>(sim.parameters.stokes_velocity_degree), dim),
      fe_levelset (sim.parameters.composition_degree)

  {
    this->initialize_simulator(sim);
    parse_parameters (prm);


  }



  template <int dim>
  void
  LevelsetHandler<dim>::declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("Levelset");

    prm.leave_subsection();
  }



  template <int dim>
  void
  LevelsetHandler<dim>::parse_parameters (ParameterHandler &prm)
  {

    prm.enter_subsection ("Levelset");
    {

    }
    prm.leave_subsection ();
  }



  template <int dim>
  void
  LevelsetHandler<dim>::initialize (ParameterHandler &/*prm*/)
  {
    AssertThrow(!this->get_material_model().is_compressible(),
                ExcMessage("The levelset method currently assumes incompressibility."));

    AssertThrow(!this->get_parameters().mesh_deformation_enabled,
                ExcMessage("The levelset method is currently incompatible with the Free Surface implementation."));

    AssertThrow(!this->get_parameters().include_melt_transport,
                ExcMessage("The levelset method has not been tested with melt transport yet, so inclusion of both is currently disabled."))

#if DEAL_II_VERSION_GTE(9,3,0)
    dof_handler_levelset.reinit(sim.triangulation);
    dof_handler_v.reinit(sim.triangulation);
#endif
  }

  template <int dim>
  void
  LevelsetHandler<dim>::setup_dofs ()
  {
    this->get_pcout() << "LevelsetHandler::setup_dofs()..." << std::endl;

    dof_handler_v.clear();

#if DEAL_II_VERSION_GTE(9,3,0)
    dof_handler_v.distribute_dofs(fe_v);
#else
    dof_handler_v.initialize(sim.triangulation, fe_v);
#endif

    DoFRenumbering::hierarchical(dof_handler_v);


    dof_handler_levelset.clear();
#if DEAL_II_VERSION_GTE(9,3,0)
    dof_handler_levelset.distribute_dofs(fe_levelset);
#else
    dof_handler_levelset.initialize(sim.triangulation, fe_levelset);
#endif
    DoFRenumbering::hierarchical(dof_handler_levelset);

  }

  template <int dim>
  void
  LevelsetHandler<dim>::solve(const typename Simulator<dim>::AdvectionField &advection_field)
  {
    this->get_pcout() << "solving levelset equation..." << std::endl;

    const unsigned int blockidx = advection_field.block_index(sim.introspection);
    LinearAlgebra::Vector field = sim.solution.block(blockidx);

    this->get_pcout() << "    note: "
                      << "old t=" << this->get_time()
                      << " dt=" << this->get_timestep()
                      << " composition component index=" << advection_field.component_index(sim.introspection)
                      << std::endl;

    // TODO: update field

    sim.solution.block(blockidx) = field;

    // In the first timestep we might want to initialize all vectors with the
    // reinitialized levelset
    if (sim.timestep_number == 0)
      {
        // TODO
        //sim.old_solution.block(blockidx) = ...
        // sim.old_old_solution.block(blockidx) = ...
      }

  }

}



namespace aspect
{
#define INSTANTIATE(dim) \
  template class LevelsetHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
