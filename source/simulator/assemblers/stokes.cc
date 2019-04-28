/*
  Copyright (C) 2016 - 2019 by the authors of the ASPECT code.

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

#include <aspect/simulator/assemblers/stokes.h>
#include <aspect/simulator.h>
#include <aspect/utilities.h>

#include <deal.II/base/signaling_nan.h>

namespace aspect
{
  namespace Assemblers
  {
    template <int dim>
    void
    StokesPreconditioner<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesPreconditioner<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesPreconditioner<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesPreconditioner<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesPreconditioner<dim>& > (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points           = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();
      const bool assemble_A_approximation = !this->get_parameters().use_full_A_block_preconditioner;

      // First loop over all dofs and find those that are in the Stokes system
      // save the component (pressure and dim velocities) each belongs to.
      for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
        {
          if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
            {
              scratch.dof_component_indices[i_stokes] = fe.system_to_component_index(i).first;
              ++i_stokes;
            }
          ++i;
        }

      // Loop over all quadrature points and assemble their contributions to
      // the preconditioner matrix
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  if (assemble_A_approximation)
                    scratch.grads_phi_u[i_stokes] =
                      scratch.finite_element_values[introspection.extractors
                                                    .velocities].symmetric_gradient(i, q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection
                                                                          .extractors.pressure].value(i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const double eta = scratch.material_model_outputs.viscosities[q];
          const double one_over_eta = 1. / eta;

          const double JxW = scratch.finite_element_values.JxW(q);

          if (assemble_A_approximation)
            {
              for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
                  if (scratch.dof_component_indices[i] ==
                      scratch.dof_component_indices[j])
                    {
                      data.local_matrix(i, j) += ((2.0 * eta * (scratch.grads_phi_u[i]
                                                                * scratch.grads_phi_u[j]))
                                                  + one_over_eta * pressure_scaling
                                                  * pressure_scaling
                                                  * (scratch.phi_p[i]
                                                     * scratch.phi_p[j]))
                                                 * JxW;
                    }

            }
          else
            {
              const unsigned int pressure_component_index = this->introspection().component_indices.pressure;
              for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                if (scratch.dof_component_indices[i] == pressure_component_index)
                  for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
                    if (scratch.dof_component_indices[j] == pressure_component_index)
                      {
                        data.local_matrix(i, j) += (one_over_eta * pressure_scaling
                                                    * pressure_scaling
                                                    * (scratch.phi_p[i]
                                                       * scratch.phi_p[j]))
                                                   * JxW;
                      }
            }
        }
    }



    template <int dim>
    void
    StokesCompressiblePreconditioner<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesPreconditioner<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesPreconditioner<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesPreconditioner<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesPreconditioner<dim>& > (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points           = scratch.finite_element_values.n_quadrature_points;

      // First loop over all dofs and find those that are in the Stokes system
      // save the component (pressure and dim velocities) each belongs to.
      for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
        {
          if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
            {
              scratch.dof_component_indices[i_stokes] = fe.system_to_component_index(i).first;
              ++i_stokes;
            }
          ++i;
        }

      // Loop over all quadrature points and assemble their contributions to
      // the preconditioner matrix
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                  scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);

                  ++i_stokes;
                }
              ++i;
            }

          const double eta_two_thirds = scratch.material_model_outputs.viscosities[q] * 2.0 / 3.0;

          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
            for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
              if (scratch.dof_component_indices[i] ==
                  scratch.dof_component_indices[j])
                data.local_matrix(i, j) += (- eta_two_thirds * (scratch.div_phi_u[i] * scratch.div_phi_u[j])
                                           )
                                           * JxW;
        }
    }



    template <int dim>
    void
    StokesIncompressibleTerms<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      const MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>
      *force = scratch.material_model_outputs.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim> >();

      const MaterialModel::ElasticOutputs<dim>
      *elastic_outputs = scratch.material_model_outputs.template get_additional_output<MaterialModel::ElasticOutputs<dim> >();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].value (i,q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  if (scratch.rebuild_stokes_matrix)
                    {
                      scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                      scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);
                    }
                  ++i_stokes;
                }
              ++i;
            }


          // Viscosity scalar
          const double eta = (scratch.rebuild_stokes_matrix
                              ?
                              scratch.material_model_outputs.viscosities[q]
                              :
                              numbers::signaling_nan<double>());

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));

          const double density = scratch.material_model_outputs.densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            {
              data.local_rhs(i) += (density * gravity * scratch.phi_u[i])
                                   * JxW;

              if (force != nullptr && this->get_parameters().enable_additional_stokes_rhs)
                data.local_rhs(i) += (force->rhs_u[q] * scratch.phi_u[i]
                                      + pressure_scaling * force->rhs_p[q] * scratch.phi_p[i])
                                     * JxW;

              if (elastic_outputs != nullptr && this->get_parameters().enable_elasticity)
                data.local_rhs(i) += (scalar_product(elastic_outputs->elastic_force[q],Tensor<2,dim>(scratch.grads_phi_u[i])))
                                     * JxW;

              if (scratch.rebuild_stokes_matrix)
                for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
                  {
                    data.local_matrix(i,j) += ( (eta * 2.0 * (scratch.grads_phi_u[i] * scratch.grads_phi_u[j]))
                                                // assemble \nabla p as -(p, div v):
                                                - (pressure_scaling *
                                                   scratch.div_phi_u[i] * scratch.phi_p[j])
                                                // assemble the term -div(u) as -(div u, q).
                                                // Note the negative sign to make this
                                                // operator adjoint to the grad p term:
                                                - (pressure_scaling *
                                                   scratch.phi_p[i] * scratch.div_phi_u[j]))
                                              * JxW;
                  }
            }
        }
    }



    template <int dim>
    void
    StokesIncompressibleTerms<dim>::
    create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const
    {
      const unsigned int n_points = outputs.viscosities.size();

      if (this->get_parameters().enable_additional_stokes_rhs
          && outputs.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim> >() == nullptr)
        {
          outputs.additional_outputs.push_back(
            std::make_shared<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>> (n_points));
        }

      Assert(!this->get_parameters().enable_additional_stokes_rhs
             ||
             outputs.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim> >()->rhs_u.size()
             == n_points, ExcInternalError());

      if ((this->get_parameters().enable_elasticity) &&
          outputs.template get_additional_output<MaterialModel::ElasticOutputs<dim> >() == nullptr)
        {
          outputs.additional_outputs.push_back(
            std::make_shared<MaterialModel::ElasticOutputs<dim>> (n_points));
        }

      Assert(!this->get_parameters().enable_elasticity
             ||
             outputs.template get_additional_output<MaterialModel::ElasticOutputs<dim> >()->elastic_force.size()
             == n_points, ExcInternalError());
    }



    template <int dim>
    void
    StokesCompressibleStrainRateViscosityTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      if (!scratch.rebuild_stokes_matrix)
        return;

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                  scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);

                  ++i_stokes;
                }
              ++i;
            }

          // Viscosity scalar
          const double eta_two_thirds = scratch.material_model_outputs.viscosities[q] * 2.0 / 3.0;

          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
              {
                data.local_matrix(i,j) += (- (eta_two_thirds * (scratch.div_phi_u[i] * scratch.div_phi_u[j])
                                             ))
                                          * JxW;
              }
        }
    }



    template <int dim>
    void
    StokesReferenceDensityCompressibilityTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      // assemble RHS of:
      //  - div u = 1/rho * drho/dz g/||g||* u
      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::reference_density_profile,
             ExcInternalError());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));
          const double drho_dz_u = scratch.reference_densities_depth_derivative[q]
                                   * (gravity * scratch.velocity_values[q]) / gravity.norm();
          const double one_over_rho = 1.0/scratch.reference_densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            data.local_rhs(i) += (pressure_scaling *
                                  one_over_rho * drho_dz_u * scratch.phi_p[i])
                                 * JxW;
        }
    }



    template <int dim>
    void
    StokesImplicitReferenceDensityCompressibilityTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      // assemble compressibility term of:
      //  - div u - 1/rho * drho/dz g/||g||* u = 0
      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::implicit_reference_density_profile,
             ExcInternalError());

      if (!scratch.rebuild_stokes_matrix)
        return;

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].value (i,q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i,q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));
          const Tensor<1,dim> drho_dz = scratch.reference_densities_depth_derivative[q]
                                        * gravity / gravity.norm();
          const double one_over_rho = 1.0/scratch.reference_densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
              data.local_matrix(i,j) += -(pressure_scaling *
                                          one_over_rho * drho_dz * scratch.phi_u[j] * scratch.phi_p[i])
                                        * JxW;
        }
    }



    template <int dim>
    void
    StokesIsothermalCompressionTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      // assemble RHS of:
      //  - div \mathbf{u} = \frac{1}{\rho} \frac{\partial rho}{\partial p} \rho \mathbf{g} \cdot \mathbf{u}

      // Compared to the manual, this term seems to have the wrong sign, but
      // this is because we negate the entire equation to make sure we get
      // -div(u) as the adjoint operator of grad(p)

      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::isothermal_compression,
             ExcInternalError());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));

          const double compressibility
            = scratch.material_model_outputs.compressibilities[q];

          const double density = scratch.material_model_outputs.densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            data.local_rhs(i) += (
                                   (pressure_scaling *
                                    compressibility * density *
                                    (scratch.velocity_values[q] * gravity) *
                                    scratch.phi_p[i])
                                 )
                                 * JxW;
        }
    }


    template <int dim>
    void
    StokesHydrostaticCompressionTerm<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      // assemble RHS of:
      // $ -\nabla \cdot \mathbf{u} = \left( \kappa \rho \textbf{g} - \alpha \nabla T \right) \cdot \textbf{u}$
      //
      // where $\frac{1}{\rho} \frac{\partial \rho}{\partial p} = \kappa$ is the compressibility,
      // $- \frac{1}{\rho}\frac{\partial \rho}{\partial T} = \alpha$ is the thermal expansion coefficient,
      // and both are defined in the material model.

      // Compared to the manual, this term seems to have the wrong sign, but
      // this is because we negate the entire equation to make sure we get
      // -div(u) as the adjoint operator of grad(p)

      Assert(this->get_parameters().formulation_mass_conservation ==
             Parameters<dim>::Formulation::MassConservation::hydrostatic_compression,
             ExcInternalError());

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));

          const double compressibility
            = scratch.material_model_outputs.compressibilities[q];

          const double thermal_alpha
            = scratch.material_model_outputs.thermal_expansion_coefficients[q];

          const double density = scratch.material_model_outputs.densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            data.local_rhs(i) += (
                                   (pressure_scaling *
                                    (
                                      // pressure part:
                                      compressibility * density *
                                      (scratch.velocity_values[q] * gravity)
                                      // temperature part:
                                      - thermal_alpha *
                                      (scratch.velocity_values[q] * scratch.temperature_gradients[q])
                                    ) * scratch.phi_p[i])
                                 )
                                 * JxW;

        }
    }


    template <int dim>
    void
    StokesPressureRHSCompatibilityModification<dim>::execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                                                              internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = scratch.finite_element_values.get_fe();

      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;

      for (unsigned int q=0; q<n_q_points; ++q)
        for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
          {
            if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
              {
                scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                data.local_pressure_shape_function_integrals(i_stokes) += scratch.phi_p[i_stokes] * scratch.finite_element_values.JxW(q);
                ++i_stokes;
              }
            ++i;
          }
    }



    template <int dim>
    void
    StokesBoundaryTraction<dim>::execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                                          internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = scratch.finite_element_values.get_fe();

      // see if any of the faces are traction boundaries for which
      // we need to assemble force terms for the right hand side
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();

      const typename DoFHandler<dim>::face_iterator face = scratch.cell->face(scratch.face_number);

      if (this->get_boundary_traction()
          .find (face->boundary_id())
          !=
          this->get_boundary_traction().end())
        {
          for (unsigned int q=0; q<scratch.face_finite_element_values.n_quadrature_points; ++q)
            {
              const Tensor<1,dim> traction
                = this->get_boundary_traction().find(
                    face->boundary_id()
                  )->second
                  ->boundary_traction (face->boundary_id(),
                                       scratch.face_finite_element_values.quadrature_point(q),
                                       scratch.face_finite_element_values.normal_vector(q));

              for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
                {
                  if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                    {
                      data.local_rhs(i_stokes) += scratch.face_finite_element_values[introspection.extractors.velocities].value(i,q) *
                                                  traction *
                                                  scratch.face_finite_element_values.JxW(q);
                      ++i_stokes;
                    }
                  ++i;
                }
            }
        }
    }


    namespace
    {
      Tensor<2,2>
      hollow_sphere_velocity_gradient (const Point<2> &pos,
                                       const double mmm)
      {
        (void)pos;
        (void)mmm;
        return Tensor<2,2>();
      }
      Tensor<2,3>
      hollow_sphere_velocity_gradient (const Point<3> &pos,
                                       const double mmm)
      {
        const double gammma = 1.0;

        const double R1 = 0.5;
        const double R2 = 1.0;


        const std::array<double,3> spos =
          aspect::Utilities::Coordinates::cartesian_to_spherical_coordinates(pos);

        const double r=spos[0];
        const double  phi=spos[1];
        const double theta=spos[2];


        double alpha,beta,fr,gr,dr_gr,dr_fr;

        if (mmm == -1)
          {
            alpha=-gammma*(pow(R2,3)-pow(R1,3))/(pow(R2,3)*log(R1)-pow(R1,3)*log(R2));
            beta=-3*gammma*(log(R2)-log(R1))/(pow(R1,3)*log(R2)-pow(R2,3)*log(R1)) ;


            fr=alpha/(r*r)+beta*r;
            gr=-2/(r*r)*(alpha*log(r)+beta/3*pow(r,3)+gammma);

            dr_fr = -2*alpha/pow(r,3)+beta;
            dr_gr = 4/pow(r,3)*(alpha*log(r)+beta/3*pow(r,3)+gammma) - 2/(r*r)*(alpha/r+beta*r*r);
          }
        else
          {
            //AssertThrow(false,ExcNotImplemented());
            alpha=gammma*(mmm+1)*(pow(R1,-3)-pow(R2,-3))/(pow(R1,-mmm-4)-pow(R2,-mmm-4));
            beta=-3*gammma*(pow(R1,mmm+1)-pow(R2,mmm+1))/(pow(R1,mmm+4)-pow(R2,mmm+4));

            fr=alpha/pow(r,mmm+3)+beta*r;
            gr=-2/(r*r)*(-alpha/(mmm+1)*pow(r,-mmm-1)+beta/3*pow(r,3)+gammma);

            dr_fr = -(mmm+3)*alpha/pow(r,mmm+4)+beta;
            dr_gr = 4/pow(r,3)*(-alpha/(mmm+1)*pow(r,-mmm-1)+beta/3*pow(r,3)+gammma) - 2/(r*r)*(alpha*pow(r,-mmm-2)+beta*r*r);
          }

        const double vr    =gr*cos(theta);
        const double vtheta=fr*sin(theta);
        const double vphi  =fr*sin(theta);

        const double dr_vr = cos(theta)*dr_gr;
        const double dtheta_vr = -sin(theta)*gr;
        const double dr_vtheta = sin(theta)*dr_fr;
        const double dtheta_vtheta = cos(theta)*fr;
        const double dr_vphi = dr_vtheta;
        const double dtheta_vphi = dtheta_vtheta;

        //        const double v_x=sin(theta)*cos(phi)*vr + cos(theta)*cos(phi)*vtheta-sin(phi)*vphi;
        //        const double v_y=sin(theta)*sin(phi)*vr + cos(theta)*sin(phi)*vtheta+cos(phi)*vphi;
        //        const double v_z=cos(theta)*vr - sin(theta)*vtheta;

        const double dr_vx = sin(theta)*cos(phi)*dr_vr + cos(theta)*cos(phi)*dr_vtheta - sin(phi)*dr_vphi;
        const double dtheta_vx = cos(phi)*(cos(theta)*vr + sin(theta)*dtheta_vr+cos(theta)*dtheta_vtheta-sin(theta)*vtheta)-sin(phi)*dtheta_vphi;
        const double dphi_vx = -sin(theta)*sin(phi)*vr - cos(theta)*sin(phi)*vtheta - cos(phi)*vphi;

        const double dr_vy = sin(theta)*sin(phi)*dr_vr + cos(theta)*sin(phi)*dr_vtheta + cos(phi)*dr_vphi;
        const double dtheta_vy = sin(phi)*(cos(theta)*vr + sin(theta)*dtheta_vr+cos(theta)*dtheta_vtheta-sin(theta)*vtheta)+cos(phi)*dtheta_vphi;
        const double dphi_vy = sin(theta)*cos(phi)*vr + cos(theta)*cos(phi)*vtheta - sin(phi)*vphi;

        const double dr_vz = cos(theta)*dr_vr - sin(theta)*dr_vtheta;
        const double dtheta_vz = cos(theta)*(dtheta_vr - vtheta) - sin(theta)*(vr + dtheta_vtheta);


        const double dx_vx = cos(phi)*sin(theta)*dr_vx - sin(phi)/(r*sin(theta))*dphi_vx + cos(phi)*cos(theta)/r*dtheta_vx;
        const double dy_vx = sin(phi)*sin(theta)*dr_vx + cos(phi)/(r*sin(theta))*dphi_vx + sin(phi)*cos(theta)/r*dtheta_vx;
        const double dz_vx = cos(theta)*dr_vx - sin(theta)/r*dtheta_vx;

        const double dx_vy = cos(phi)*sin(theta)*dr_vy - sin(phi)/(r*sin(theta))*dphi_vy + cos(phi)*cos(theta)/r*dtheta_vy;
        const double dy_vy = sin(phi)*sin(theta)*dr_vy + cos(phi)/(r*sin(theta))*dphi_vy + sin(phi)*cos(theta)/r*dtheta_vy;
        const double dz_vy = cos(theta)*dr_vy - sin(theta)/r*dtheta_vy;

        const double dx_vz = cos(phi)*sin(theta)*dr_vz + cos(phi)*cos(theta)/r*dtheta_vz;
        const double dy_vz = sin(phi)*sin(theta)*dr_vz + sin(phi)*cos(theta)/r*dtheta_vz;
        const double dz_vz = cos(theta)*dr_vz - sin(theta)/r*dtheta_vz;


        Tensor<2,3> gradient;
        gradient[0][0] = dx_vx;
        gradient[0][1] = (dy_vx+dx_vy)/2;
        gradient[0][2] = (dz_vx+dx_vz)/2;
        gradient[1][0] = (dx_vy+dy_vx)/2;
        gradient[1][1] = dy_vy;
        gradient[1][2] = (dz_vy+dy_vz)/2;
        gradient[2][0] = (dx_vz+dz_vx)/2;
        gradient[2][1] = (dy_vz+dz_vy)/2;
        gradient[2][2] = dz_vz;

        //double gradient_norm = gradient.norm();
        //if (gradient_norm > 1e10)
        //std::cout << "(r,theta,phi) = (" << r << "," << theta << "," << phi << ") " << std::endl;
        //   std::cout << gradient.norm() << std::endl;

        return gradient;
      }

      template<int dim>
      Tensor<1,dim>
      tensor_vmult(const Tensor<2,dim> mat,const Tensor<1,dim> vec)
      {
        Tensor<1,dim> return_vec;
        for (unsigned int i=0; i<dim; ++i)
          {
            double val = 0;
            for (unsigned int j=0; j<dim; ++j)
              val += mat[i][j]*vec[j];
            return_vec[i] = val;
          }
        return return_vec;
      }
    }
    template <int dim>
    void
    StokesBoundaryNEW<dim>::execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                                     internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = scratch.finite_element_values.get_fe();

      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();

      const typename DoFHandler<dim>::face_iterator face = scratch.cell->face(scratch.face_number);

      if (face->boundary_id() == 0)
        {
          for (unsigned int q=0; q<scratch.face_finite_element_values.n_quadrature_points; ++q)
            {
              double eta = 2.0;//scratch.material_model_outputs.viscosities[q];
              //std::cout << scratch.material_model_outputs.viscosities[q] << std::endl;

              const Tensor<2,dim> gradient = 2.0*eta*hollow_sphere_velocity_gradient(scratch.face_finite_element_values.quadrature_point(q),-2);
              const Tensor<1,dim> grad_dot_n = gradient*scratch.face_finite_element_values.normal_vector(q);
              const Tensor<1,dim> grad_dot_n_X_n = cross_product_3d(grad_dot_n, scratch.face_finite_element_values.normal_vector(q));

              for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
                {
                  if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                    {
                      const Tensor<1,dim> shape_value(scratch.face_finite_element_values[introspection.extractors.velocities].value(i,q));
                      const Tensor<1,dim> vXn = cross_product_3d(shape_value,
                                                                 scratch.face_finite_element_values.normal_vector(q));

                      data.local_rhs(i_stokes) += scalar_product(grad_dot_n_X_n,vXn)
                                                  *scratch.face_finite_element_values.JxW(q);
                      ++i_stokes;
                    }
                  ++i;
                }
            }
        }
    }
  }
} // namespace aspect

// explicit instantiation of the functions we implement in this file
namespace aspect
{
  namespace Assemblers
  {
#define INSTANTIATE(dim) \
  template class StokesPreconditioner<dim>; \
  template class StokesCompressiblePreconditioner<dim>; \
  template class StokesIncompressibleTerms<dim>; \
  template class StokesCompressibleStrainRateViscosityTerm<dim>; \
  template class StokesReferenceDensityCompressibilityTerm<dim>; \
  template class StokesImplicitReferenceDensityCompressibilityTerm<dim>; \
  template class StokesIsothermalCompressionTerm<dim>; \
  template class StokesHydrostaticCompressionTerm<dim>; \
  template class StokesPressureRHSCompatibilityModification<dim>; \
  template class StokesBoundaryTraction<dim>; \
  template class StokesBoundaryNEW<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)
  }
}
