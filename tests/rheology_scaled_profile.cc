/*
  Copyright (C) 2019 - 2020 by the authors of the ASPECT code.

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

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/global.h>
#include <aspect/adiabatic_conditions/interface.h>

#include <aspect/geometry_model/interface.h>
#include <aspect/material_model/rheology/ascii_depth_profile.h>
#include <aspect/lateral_averaging.h>

namespace aspect
{
  using namespace dealii;

  namespace MaterialModel
  {
    /**
     * Additional output fields for the viscosity prior to scaling to be added to
     * the MaterialModel::MaterialModelOutputs structure and filled in the
     * MaterialModel::Interface::evaluate() function.
     */
    template <int dim>
    class UnscaledViscosityAdditionalOutputs : public NamedAdditionalMaterialOutputs<dim>
    {
      public:
        UnscaledViscosityAdditionalOutputs(const unsigned int n_points)
          : NamedAdditionalMaterialOutputs<dim>(std::vector<std::string>(1, "unscaled_viscosity"),
                                                n_points)
        {}
    };
  }

  namespace internal
  {
    template <int dim>
    class FunctorDepthAverageUnscaledViscosity: public internal::FunctorBase<dim>
    {
      public:
        FunctorDepthAverageUnscaledViscosity()
        {}

        bool need_material_properties() const override
        {
          return true;
        }

        void
        create_additional_material_model_outputs (const unsigned int n_points,
                                                  MaterialModel::MaterialModelOutputs<dim> &outputs) const override
        {
          if (outputs.template get_additional_output<MaterialModel::UnscaledViscosityAdditionalOutputs<dim>>() == nullptr)
            {
              outputs.additional_outputs.push_back(
                std_cxx14::make_unique<MaterialModel::UnscaledViscosityAdditionalOutputs<dim>> (n_points));
            }
        }

        void operator()(const MaterialModel::MaterialModelInputs<dim> &,
                        const MaterialModel::MaterialModelOutputs<dim> &out,
                        const FEValues<dim> &,
                        const LinearAlgebra::BlockVector &,
                        std::vector<double> &output) override
        {
          const MaterialModel::UnscaledViscosityAdditionalOutputs<dim> *unscaled_viscosity_outputs
            = out.template get_additional_output<const MaterialModel::UnscaledViscosityAdditionalOutputs<dim> >();

          Assert(unscaled_viscosity_outputs != nullptr,ExcInternalError());

          for (unsigned int q=0; q<output.size(); ++q)
            output[q] = unscaled_viscosity_outputs->output_values[0][q];
        }
    };
  }

  namespace MaterialModel
  {
    /**
     * A material model that tests the variable depth averaging routines. The material properties are
     * constant, except that the viscosity is set to a reference viscosity profile by first computing
     * an unscaled (constant) viscosity profile, which is then scaled by the quotient between reference
     * profile and unscaled viscosity.
     */
    template<int dim>
    class ScaledViscosityProfileMaterial : public MaterialModel::Interface<dim>, public aspect::SimulatorAccess<dim>
    {
      public:
        void initialize() override
        {
          reference_viscosity_coordinates = reference_viscosity_profile->get_coordinates();
        }

        void update() override // only happens before the temperature system is solved
        {
          std::vector<std::unique_ptr<internal::FunctorBase<dim> > > lateral_averaging_properties;
          lateral_averaging_properties.push_back(std::make_unique<internal::FunctorDepthAverageUnscaledViscosity<dim>>());

          std::vector<std::vector<double>> averages =
                                          this->get_lateral_averaging().compute_lateral_averages(reference_viscosity_coordinates,
                                              lateral_averaging_properties);

          laterally_averaged_viscosity_profile.swap(averages[0]);

          for (unsigned int i = 0; i < laterally_averaged_viscosity_profile.size(); ++i)
            AssertThrow(numbers::is_finite(laterally_averaged_viscosity_profile[i]),
                        ExcMessage("In computing depth averages, there is at"
                                   " least one depth band that does not have"
                                   " any quadrature points in it."
                                   " Consider reducing number of depth layers"
                                   " for averaging."));
          initialized = true;
        }



        double
        compute_viscosity_scaling (const double depth) const
        {
          // Make maximal depth slightly larger to ensure depth < maximal_depth
          const double maximal_depth = this->get_geometry_model().maximal_depth() *
                                       (1.0+std::numeric_limits<double>::epsilon());
          Assert(depth < maximal_depth, ExcInternalError());

          unsigned int depth_index;
          if (depth < reference_viscosity_coordinates.front())
            {
              depth_index = 0;
            }
          else if (depth > reference_viscosity_coordinates.back())
            {
              depth_index = reference_viscosity_coordinates.size() - 1;
            }
          else
            {
              depth_index = std::distance(reference_viscosity_coordinates.begin(),
                                          std::lower_bound(reference_viscosity_coordinates.begin(),
                                                           reference_viscosity_coordinates.end(),
                                                           depth));
            }
          // depth_index calculated above is the first layer boundary larger than depth,
          // the correct layer index is then one less than this (except for depth == depth_bounds[0],
          // in which case the depth_bounds index is also the layer_index, namely 0).
          // For depths > depth_bounds[n-1], where, n is the size of the depth_bounds, we need to use
          // the depth_bound one less than this, i.e. depth_bounds[n-2] to compute the averages of the
          // last layer.
          if (depth_index > 0)
            depth_index -= 1;

          // When evaluating reference viscosity, evaluate at the next lower depth that is stored
          // in the reference profile instead of the actual depth. This makes the profile piecewise
          // constant. This will be specific to the viscosity profile used (and ignore the entry with
          // the largest depth in the profile).
          const double reference_viscosity = reference_viscosity_profile->compute_viscosity(reference_viscosity_coordinates.at(depth_index));

          const double average_viscosity = std::pow(10, laterally_averaged_viscosity_profile[depth_index]);

          return reference_viscosity / average_viscosity;
        }



        void compute_equilibrium_grain_size(const typename Interface<dim>::MaterialModelInputs &in,
                                            typename Interface<dim>::MaterialModelOutputs &out) const
        {
          PrescribedFieldOutputs<dim> *grain_size_out = out.template get_additional_output<PrescribedFieldOutputs<dim> >();
          // DislocationViscosityOutputs<dim> *disl_viscosities_out = out.template get_additional_output<DislocationViscosityOutputs<dim> >();

          UnscaledViscosityAdditionalOutputs<dim> *unscaled_viscosity_out =
            out.template get_additional_output<MaterialModel::UnscaledViscosityAdditionalOutputs<dim> >();

          for (unsigned int i=0; i<in.position.size(); ++i)
            {
              // get the dislocation viscosity to compute the dislocation strain rate
              // If we do not have the strain rate yet, set it to a low value.
              // TODO: make minimum strain rate an input parameter
              const SymmetricTensor<2,dim> shear_strain_rate = in.strain_rate[i] - 1./dim * trace(in.strain_rate[i]) * unit_symmetric_tensor<dim>();
              const double second_strain_rate_invariant = std::sqrt(std::abs(second_invariant(shear_strain_rate)));

              // TODO: if we update the interface of the diffusion_viscosity and dislocation_viscosity functions,
              // we don't need this vector anymore
              std::vector<double> composition (in.composition[i]);
              double grain_size = 1e-3;

              double diff_viscosity;
              double effective_viscosity = 1e20;
              double disl_viscosity = effective_viscosity;
              out.viscosities[i] = effective_viscosity;

              if ( (unscaled_viscosity_out != nullptr) )
                unscaled_viscosity_out->output_values[0][i] = std::log10(out.viscosities[i]);

              // If we do not have the strain rate, there is no equilibrium grain size
              if (second_strain_rate_invariant > 1e-30)
                {
                  unsigned int j = 0;
                  double old_grain_size = 0.0;

                  //  only goes in this loop during post-processing 

                  // because the diffusion viscosity depends on the grain size itself, and we need it to compute the dislocation strain rate,
                  // we have to iterate in the computation of the equilibrium grain size
                  while ((std::abs((grain_size-old_grain_size) / grain_size) > 1e-3) && (j < 100))
                    {
                      diff_viscosity = 1e24;
                      disl_viscosity = 1e22;

                      effective_viscosity = diff_viscosity;
                      if (std::abs(second_strain_rate_invariant) > 1e-30)
                        effective_viscosity = diff_viscosity; // * disl_viscosity / (disl_viscosity + diff_viscosity);

                      old_grain_size = grain_size;

                      ++j;
                    }

                  out.viscosities[i] = effective_viscosity;// 1e26*std::pow(100, spherical_coordinates[1]/numbers::PI) ;

                  if ( (unscaled_viscosity_out != nullptr) )
                    unscaled_viscosity_out->output_values[0][i] = std::log10(out.viscosities[i]);
                  if (this->simulator_is_past_initialization() == true && initialized) // using old out.viscosities
                    {
                      out.viscosities[i] *= compute_viscosity_scaling(this->get_geometry_model().depth(in.position[i]));
                    }
                }

              // effective visocisty is 1e20 and then 1e24 before and after the postprocessing steps, respectively,
              // but the average viscosity is 1e20 and the scaling are computed once using this value.

              if (grain_size_out != NULL)
                for (unsigned int c=0; c<composition.size(); ++c)
                  grain_size_out->prescribed_field_outputs[i][c] = 1e-3;
            }
          return;
        }


        void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                      MaterialModel::MaterialModelOutputs<dim> &out) const override
        {
          for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
            {
              out.densities[i] = 3300.0;

              if (in.strain_rate.size() > 0) // this is 0 in solving temperature system
                {
                  compute_equilibrium_grain_size(in, out);
                }

              out.compressibilities[i] = 0;
              out.specific_heat[i] = 1200;
              out.thermal_expansion_coefficients[i] = 3e-5;
              out.thermal_conductivities[i] = 4.7;

            }
        }


        bool is_compressible() const override
        {
          return false;
        }

        double reference_viscosity() const override
        {
          return 1;
        }

        void
        create_additional_named_outputs (MaterialModelOutputs<dim> &outputs) const override
        {
          if (outputs.template get_additional_output<UnscaledViscosityAdditionalOutputs<dim>>() == nullptr)
            {
              const unsigned int n_points = outputs.n_evaluation_points();
              outputs.additional_outputs.push_back(
                std_cxx14::make_unique<UnscaledViscosityAdditionalOutputs<dim>> (n_points));
            }
        }

        static
        void
        declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Material model");
          {
            // Depth-dependent parameters from the rheology plugin
            Rheology::AsciiDepthProfile<dim>::declare_parameters(prm, "Depth dependent model");
          }
          prm.leave_subsection();
        }

        void
        parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Material model");
          {
            reference_viscosity_profile = std_cxx14::make_unique<Rheology::AsciiDepthProfile<dim>>();
            reference_viscosity_profile->initialize_simulator (this->get_simulator());
            reference_viscosity_profile->parse_parameters(prm, "Depth dependent model");
            reference_viscosity_profile->initialize();
          }
          prm.leave_subsection();

          this->model_dependence.viscosity = MaterialModel::NonlinearDependence::none;
          this->model_dependence.density = MaterialModel::NonlinearDependence::none;
          this->model_dependence.compressibility = MaterialModel::NonlinearDependence::none;
          this->model_dependence.specific_heat = MaterialModel::NonlinearDependence::none;
          this->model_dependence.thermal_conductivity = MaterialModel::NonlinearDependence::none;
        }

      private:
        /**
         * A reference depth-profile for viscosity. Can be used for simply
         * depth-dependent rheologies, or to compare (and potentially scale)
         * the grain-size dependent viscosity to a reference profile.
         */
        std::unique_ptr<Rheology::AsciiDepthProfile<dim> > reference_viscosity_profile;

        /**
         * Reference viscosity profile coordinates.
         */
        std::vector<double> reference_viscosity_coordinates;

        /**
         * A depth-profile of the laterally averaged viscosity in each layer
         * in the current model. Can be used to compare (and potentially scale)
         * the computed viscosity to the reference profile.
         */
        std::vector<double> laterally_averaged_viscosity_profile;

        bool initialized;
    };
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(ScaledViscosityProfileMaterial,
                                   "scaled viscosity profile",
                                   "A material model that scales a constant viscosity to a given "
                                   "reference profile.")

#define INSTANTIATE(dim) \
  template class UnscaledViscosityAdditionalOutputs<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

  }
}
