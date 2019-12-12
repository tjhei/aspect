/*
  Copyright (C) 2011 - 2019 by the authors of the ASPECT code.

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


#include <aspect/postprocess/visualization/averaged_strain_rate.h>



namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {


      template <int dim>
      AveragedStrainRate<dim>::AveragedStrainRate()
      {}

      template <int dim>
      std::pair<std::string, Vector<float> *>
      AveragedStrainRate<dim>::execute() const
      {
        return std::make_pair(std::string("bla"), nullptr);
      }

      template <int dim>
      std::pair<std::string, Vector<float> *>
      AveragedStrainRate<dim>::      compute (const unsigned int quadrature_degree, const bool compressible, const std::string &name) const
      {
        std::pair<std::string, Vector<float> *>
        return_value (name,
                      new Vector<float>(this->get_triangulation().n_active_cells()));

        const QMidpoint<1> quadrature_formula_mp;
        const QIterated<dim> quadrature_formula(quadrature_formula_mp, quadrature_degree);
        const unsigned int n_q_points = quadrature_formula.size(); // this is 1 for QMidpoint

        FEValues<dim> fe_values(this->get_mapping(), this->get_fe(),
                                quadrature_formula,
                                update_values | update_gradients | update_quadrature_points | update_JxW_values);

        // Set up cell iterator for looping
        typename DoFHandler<dim>::active_cell_iterator
        cell = this->get_dof_handler().begin_active(),
        endc = this->get_dof_handler().end();


        std::vector<SymmetricTensor<2,dim> > strain_rates(n_q_points);

        // Loop over cells and calculate theta in each one
        // Note that we start after timestep 0 because we need the strain rate,
        // which doesn't exist during the initial step
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                fe_values[this->introspection().extractors.velocities].get_function_symmetric_gradients (this->get_solution(), strain_rates);



                double area = 0.0;
                double incompressible_strain_rate_value = 0.0;
                double compressible_strain_rate_value = 0.0;

                for (unsigned int q=0; q<n_q_points; ++q)
                  {
                    const SymmetricTensor<2,dim> strain_rate = strain_rates[q];

                    incompressible_strain_rate_value +=
                      std::sqrt(strain_rate * strain_rate)
                      * fe_values.JxW(q);

                    const SymmetricTensor<2,dim> compressible_strain_rate
                      = (this->get_material_model().is_compressible()
                         ?
                         strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                         :
                         strain_rate);

                    compressible_strain_rate_value += std::sqrt(compressible_strain_rate*compressible_strain_rate)
                                                      * fe_values.JxW(q);

                    area += fe_values.JxW(q);

                  }

                (*return_value.second)(cell->active_cell_index()) = (compressible?compressible_strain_rate_value:incompressible_strain_rate_value)/area;
              }

          }


        return return_value;
      }


      template <int dim>
      std::list<std::pair<std::string, Vector<float> *>>
                                                      AveragedStrainRate<dim>::      execute2 () const
      {
        std::list<std::pair<std::string, Vector<float> *>> return_value;
        return_value.push_back(this->compute(1, true, "asr_dev_midp"));
        return_value.push_back(this->compute(1, false, "asr_incompr_midp"));
        return_value.push_back(this->compute(5, true, "asr_dev_avg"));
        return_value.push_back(this->compute(5, false, "asr_incompr_avg"));

        return return_value;
      }

//      template <int dim>
//      void
//      StrainRate<dim>::
//      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
//                            std::vector<Vector<double> > &computed_quantities) const
//      {
//        const unsigned int n_quadrature_points = input_data.solution_values.size();
//        Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
//        Assert (computed_quantities[0].size() == 1,                   ExcInternalError());
//        Assert (input_data.solution_values[0].size() == this->introspection().n_components,           ExcInternalError());
//        Assert (input_data.solution_gradients[0].size() == this->introspection().n_components,          ExcInternalError());

//        for (unsigned int q=0; q<n_quadrature_points; ++q)
//          {
//            // extract the primal variables
//            Tensor<2,dim> grad_u;
//            for (unsigned int d=0; d<dim; ++d)
//              grad_u[d] = input_data.solution_gradients[q][d];

//            const SymmetricTensor<2,dim> strain_rate = symmetrize (grad_u);
//            const SymmetricTensor<2,dim> compressible_strain_rate
//              = (this->get_material_model().is_compressible()
//                 ?
//                 strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
//                 :
//                 strain_rate);
//            computed_quantities[q](0) = std::sqrt(compressible_strain_rate *
//                                                  compressible_strain_rate);
//          }
//      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(AveragedStrainRate,
                                                  "averaged strain rate",
                                                  "A visualization output object that generates output "
                                                  "for the norm of the strain rate, i.e., for the quantity "
                                                  "$\\sqrt{\\varepsilon(\\mathbf u):\\varepsilon(\\mathbf u)}$ "
                                                  "in the incompressible case and "
                                                  "$\\sqrt{[\\varepsilon(\\mathbf u)-\\tfrac 13(\\textrm{tr}\\;\\varepsilon(\\mathbf u))\\mathbf I]:"
                                                  "[\\varepsilon(\\mathbf u)-\\tfrac 13(\\textrm{tr}\\;\\varepsilon(\\mathbf u))\\mathbf I]}$ "
                                                  "in the compressible case.")
    }
  }
}
