/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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


#include <aspect/introspection.h>
#include <aspect/global.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/base/std_cxx1x/tuple.h>

namespace aspect
{
  namespace internal
  {
    /**
     * Return a filled ComponentIndices structure.
     */
    template <int dim>
    typename Introspection<dim>::ComponentIndices
    setup_component_indices (dealii::IntrospectionBase<dim> &ib)
    {
      typename Introspection<dim>::ComponentIndices ci;
      for (unsigned int i=0; i<dim; ++i)
        ci.velocities[i] = ib.variable("velocity").first_component_index+i;

      ci.pressure = ib.variable("pressure").first_component_index;
      ci.temperature = ib.variable("temperature").first_component_index;
      unsigned int n_compositional_fields = ib.variable("compositions").n_components();
      for (unsigned int i=0; i<n_compositional_fields; ++i)
        ci.compositional_fields.push_back(ib.variable("compositions").first_component_index+i);

      return ci;
    }

    /**
     * return filled BlockIndices structure.
     */
    template <int dim>
    typename Introspection<dim>::BlockIndices
    setup_blocks (dealii::IntrospectionBase<dim> &ib)
    {
      typename Introspection<dim>::BlockIndices b;
      b.velocities = ib.variable("velocity").block_index;
      b.pressure = ib.variable("pressure").block_index;
      b.temperature = ib.variable("temperature").block_index;

      unsigned int n_compositional_fields = ib.variable("compositions").n_components();
      for (unsigned int i=0; i<n_compositional_fields; ++i)
        b.compositional_fields.push_back(ib.variable("compositions").block_index+i);

      return b;
    }

    template <int dim>
    typename Introspection<dim>::BaseElements
    setup_base_elements (dealii::IntrospectionBase<dim> &ib)
    {
      typename Introspection<dim>::BaseElements base_elements;

      base_elements.velocities = ib.variable("velocity").base_index;
      base_elements.pressure = ib.variable("pressure").base_index;
      base_elements.temperature = ib.variable("temperature").base_index;
      base_elements.velocities = ib.variable("velocity").base_index;
      base_elements.compositional_fields = ib.variable("compositions").base_index;

      return base_elements;
    }

    template <int dim>
    std_cxx11::shared_ptr<FiniteElement<dim> >
    new_FE_Q_or_DGP(const bool discontinuous,
                    const unsigned int degree)
    {
      if (discontinuous)
        return std_cxx11::shared_ptr<FiniteElement<dim> >(new FE_DGP<dim>(degree));
      else
        return std_cxx11::shared_ptr<FiniteElement<dim> >(new FE_Q<dim>(degree));
    }

  }

  template <int dim>
  std::vector<dealii::Variable<dim> >
  Introspection<dim>::construct_variables (const Parameters<dim> &parameters)
  {
    std::vector<dealii::Variable<dim> > variables;
    bool use_direct_solver = parameters.use_direct_stokes_solver;


    // u
    variables.push_back(
      dealii::Variable<dim>("velocity",
                            std_cxx11::shared_ptr<FiniteElement<dim> >(
                              new FE_Q<dim>(parameters.stokes_velocity_degree)),
                            dim,
                            use_direct_solver?0:1)
    );

    if (parameters.include_melt_transport)
      {
        // p_f
        variables.push_back(
          dealii::Variable<dim>(
            "fluid pressure",
            internal::new_FE_Q_or_DGP<dim>(parameters.use_locally_conservative_discretization,
                                           parameters.stokes_velocity_degree-1),
            1,
            0));

        // p_c
        variables.push_back(
          dealii::Variable<dim>("compaction pressure",
                                internal::new_FE_Q_or_DGP<dim>(parameters.use_locally_conservative_discretization,
                                                               parameters.stokes_velocity_degree-1),
                                1,
                                1));

        // u_f
        variables.push_back(
          dealii::Variable<dim>("fluid velocity",
                                std_cxx11::shared_ptr<FiniteElement<dim> >(
                                  new FE_Q<dim>(parameters.stokes_velocity_degree)),
                                dim,
                                1));
      }

    // pressure
    variables.push_back(
      dealii::Variable<dim>(
        "pressure",
        internal::new_FE_Q_or_DGP<dim>(parameters.use_locally_conservative_discretization,
                                       parameters.stokes_velocity_degree-1),
        1,
        1));

    // T
    variables.push_back(
      dealii::Variable<dim>(
        "temperature",
        std_cxx11::shared_ptr<FiniteElement<dim> >(
          new FE_Q<dim>(parameters.temperature_degree)),
        1,
        1));

    // compositions:
    variables.push_back(
      dealii::Variable<dim>(
        "compositions",
        std_cxx11::shared_ptr<FiniteElement<dim> >(new FE_Q<dim>(parameters.composition_degree)),
        parameters.n_compositional_fields,
        parameters.n_compositional_fields));

    return variables;
  }


  template <int dim>
  Introspection<dim>::Introspection(const std::vector<dealii::Variable<dim> > &variables_,
                                    const Parameters<dim> &parameters
                                   )
    :
    dealii::IntrospectionBase<dim>(variables_),
    n_components (dealii::IntrospectionBase<dim>::n_components()),
    component_indices (internal::setup_component_indices<dim>(*this)),
    n_blocks(dealii::IntrospectionBase<dim>::n_blocks()),
    block_indices (internal::setup_blocks<dim>(*this)),
    extractors (component_indices),
    base_elements (internal::setup_base_elements<dim>(*this)),
    system_dofs_per_block (n_blocks),
    composition_names(parameters.names_of_compositional_fields)
  {
  }

  template <int dim>
  Introspection<dim>::~Introspection ()
  {}


  namespace
  {
    std::vector<FEValuesExtractors::Scalar>
    make_extractor_sequence (const std::vector<unsigned int> &compositional_fields)
    {
      std::vector<FEValuesExtractors::Scalar> x;
      for (unsigned int i=0; i<compositional_fields.size(); ++i)
        x.push_back (FEValuesExtractors::Scalar(compositional_fields[i]));
      return x;
    }
  }

  template <int dim>
  Introspection<dim>::Extractors::Extractors (const Introspection<dim>::ComponentIndices &component_indices)
    :
    velocities (component_indices.velocities[0]),
    pressure (component_indices.pressure),
    temperature (component_indices.temperature),
    compositional_fields (make_extractor_sequence (component_indices.compositional_fields))
  {}

  template <int dim>
  unsigned int
  Introspection<dim>::compositional_index_for_name (const std::string &name) const
  {
    std::vector<std::string>::const_iterator it = std::find(composition_names.begin(), composition_names.end(), name);
    if (it == composition_names.end())
      {
        AssertThrow (false, ExcMessage ("The compositional field " + name +
                                        " you asked for is not used in the simulation."));
      }
    else
      return it - composition_names.begin();
    return numbers::invalid_unsigned_int;
  }

  template <int dim>
  std::string
  Introspection<dim>::name_for_compositional_index (const unsigned int index) const
  {
    // make sure that what we get here is really an index of one of the compositional fields
    AssertIndexRange(index,composition_names.size());
    return composition_names[index];
  }

  template <int dim>
  bool
  Introspection<dim>::compositional_name_exists (const std::string &name) const
  {
    return (std::find(composition_names.begin(), composition_names.end(), name) != composition_names.end()
            ?
            true
            :
            false);
  }


}


// explicit instantiations
namespace aspect
{
#define INSTANTIATE(dim) \
  template struct Introspection<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)
}
