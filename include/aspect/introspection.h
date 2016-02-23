/*
  Copyright (C) 2011, 2012 by the authors of the ASPECT code.

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


#ifndef __aspect__introspection_h
#define __aspect__introspection_h

#include <deal.II/base/index_set.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/fe.h>

#include <aspect/parameters.h>

namespace dealii
{
  /**
   * Represent a single variable of the deal.II finite element system
   */
  template <int dim>
  struct Variable
  {
    /**
     * Constructor for a variable.
     *
     * @param name A user-friendly and unique string representation.
     *
     * @param fe The FiniteElement class to use.
     *
     * @param multiplicity Number of copies of the @p fe to create.
     *
     * @param n_blocks Number of blocks this variable represents inside the
     * linear system. A value of 0 will add the next variable to the current
     * block, 1 will put this variable into a single block. A value that is
     * euqal to the number of components will create a block for each
     * component.
     */
    Variable(const std::string &name,
             std_cxx11::shared_ptr<FiniteElement<dim> > fe,
             const unsigned int multiplicity,
             const unsigned int n_blocks)
      : name(name), fe(fe), multiplicity(multiplicity), n_blocks(n_blocks),
        first_component_index(-1), block_index(-1), base_index(-1)
    {
      Assert(n_blocks == 0
             || n_blocks == 1
             || n_blocks == n_components(),
             ExcMessage("A Variable can only have 0, 1, or n_component number of blocks"));
    }

    unsigned int n_components() const
    {
      return fe->n_components() * multiplicity;
    }

    const dealii::FEValuesExtractors::Scalar &extractor_scalar() const
    {
      Assert(n_components()==1, ExcMessage("not a scalar FE"));
      return scalar_extractor;
    }

    const dealii::FEValuesExtractors::Vector &extractor_vector() const
    {
      Assert(n_components()==dim, ExcMessage("not a vector FE"));
      return vector_extractor;
    }

    void initialize(const unsigned int component_index,
                    const unsigned int block_index_,
                    const unsigned int base_index_
                   )
    {
      block_index = block_index_;
      first_component_index = component_index;
      base_index = base_index_;
      if (n_components()==1)
        scalar_extractor.component = component_index;
      else
        scalar_extractor.component = -1;

      if (n_components()==dim)
        vector_extractor.first_vector_component = component_index;
      else
        vector_extractor.first_vector_component = -1;
    }

    std::string name;
    std_cxx11::shared_ptr<FiniteElement<dim> > fe;
    unsigned int multiplicity;
    unsigned int n_blocks;

    unsigned int first_component_index;
    unsigned int block_index;
    unsigned int base_index;

    dealii::FEValuesExtractors::Scalar scalar_extractor;
    dealii::FEValuesExtractors::Vector vector_extractor;
    ComponentMask component_mask;
  };

  template <int dim>
  class IntrospectionBase
  {
    public:
      IntrospectionBase() {}
      IntrospectionBase(const std::vector<Variable<dim> > &variables)
      {
        initialize(variables);
      }

      void initialize(const std::vector<Variable<dim> > &variables)
      {
        variables_ = variables;
        name_map.clear();

        unsigned int component_index = 0;
        unsigned int block_index = 0;

        for (unsigned int i=0; i<variables_.size(); ++i)
          {
            variables_[i].initialize(component_index, block_index, i);
            component_index+= variables_[i].n_components();
            block_index += variables_[i].n_blocks;

            name_map.insert(std::make_pair(variables_[i].name, &variables_[i]));
          }
        Assert(variables_.back().n_blocks != 0
               || variables_.back().n_components() == 0
               , ExcMessage("last variable needs to have >0 blocks"));

        n_components_ = component_index;
        n_blocks_ = block_index;

        fes.resize(variables_.size());
        multiplicities.resize(variables_.size());
        for (unsigned int i=0; i<variables_.size(); ++i)
          {
            fes[i] = &*(variables_[i].fe);
            multiplicities[i] = variables_[i].multiplicity;
            variables_[i].component_mask=ComponentMask(n_components_, false);
            for (unsigned int c=0; c<variables_[i].n_components(); ++c)
              variables_[i].component_mask.set(c+variables_[i].first_component_index, true);
          }

        components_to_blocks_.clear();
        for (unsigned int i=0; i<variables_.size(); ++i)
          {
            for (unsigned int c=0; c<variables_[i].n_components(); ++c)
              components_to_blocks_.push_back(variables_[i].block_index
                                              + ((variables_[i].n_blocks>1)?c:0));
          }
      }

      const Variable<dim> &variable(const std::string &name) const
      {
        typename std::map<std::string, Variable<dim>*>::const_iterator it
          = name_map.find(name);
        Assert(it != name_map.end(), ExcMessage("Variable '" + name + "' not found!"));
        return *(it->second);
      }

      const std::vector<Variable<dim> > variables() const
      {
        return variables_;
      }

      const unsigned int n_components() const
      {
        return n_components_;
      }
      const unsigned int n_blocks() const
      {
        return n_blocks_;
      }

      /**
       * Return the vector of finite element spaces used for the construction
       * of the FESystem.
       */
      const std::vector<const FiniteElement<dim> *> &get_fes() const
      {
        return fes;
      }

      /**
       * Return the vector of multiplicities used for the construction of the
       * FESystem.
       */
      const std::vector<unsigned int> &get_multiplicities() const
      {
        return multiplicities;
      }

      const std::vector<unsigned int> &components_to_blocks() const
      {
        return components_to_blocks_;
      }

    protected:
      std::vector<Variable<dim> > variables_;

      std::map<std::string, Variable<dim>* > name_map;
      unsigned int n_components_;
      unsigned int n_blocks_;
      std::vector<const FiniteElement<dim> *> fes;
      std::vector<unsigned int> multiplicities;
      std::vector<unsigned int> components_to_blocks_;
  };
}

namespace aspect
{
  using namespace dealii;




  /**
   * The introspection class provides information about the simulation as a
   * whole. In particular, it provides symbolic names for things like the
   * velocity, pressure and other variables, along with their corresponding
   * vector and scalar FEValues extractors, component masks, etc.
   *
   * The purpose of this class is primarily to provide these symbolic names so
   * that we do not have to use implicit knowledge about the ordering of
   * variables (e.g., earlier versions of ASPECT had many places where we
   * built a scalar FEValues extractor at component 'dim' since that is where
   * we knew that the pressure lies in the finite element; this kind of
   * implicit knowledge is no longer necessary with the Introspection class).
   * The Introspection class is used both by the Simulator class itself, but
   * is also exported to plugins via the SimulatorAccess class.
   *
   * The layout of the unknowns is the following:
   *
   * velocity pressure (in one block if using a direct solver)
   * temperature
   * composition1
   * ...
   *
   * With melt transport the layout becomes:
   * velocity fluid_pressure&compaction_pressure
   * fluid_velocity
   * pressure
   * temperature
   * composition1
   * ...
   *
   *
   * @ingroup Simulator
   */
  template <int dim>
  struct Introspection: public dealii::IntrospectionBase<dim>
  {
    public:
      /**
       * Constructor.
       * @param add_compaction_pressure Set to true if the compaction pressure
       * should be added. TODO: there are different cases for the block
       */
      Introspection (const std::vector<dealii::Variable<dim> > &variables,
                     const Parameters<dim> &parameters);

      /**
       * Destructor.
       */
      ~Introspection ();


      static std::vector<Variable<dim> >
      construct_variables (const Parameters<dim> &parameters);

      /**
       * @name Things that are independent of the current mesh
       * @{
       */
      /**
       * The number of vector components used by the finite element
       * description of this problem. It equals $d+2+n_c$ where $d$ is the
       * dimension and equals the number of velocity components, and $n_c$ is
       * the number of advected (compositional) fields. The remaining
       * components are the scalar pressure and temperature fields.
       */
      const unsigned int n_components;

      /**
       * A structure that enumerates the vector components of the finite
       * element that correspond to each of the variables in this problem.
       */
      struct ComponentIndices
      {
        unsigned int       velocities[dim];
        unsigned int       pressure;
        //unsigned int       fluid_velocities[dim];
        //unsigned int       fluid_pressure;
        //unsigned int       compaction_pressure;
        unsigned int       temperature;
        std::vector<unsigned int> compositional_fields;
      };
      /**
       * A variable that enumerates the vector components of the finite
       * element that correspond to each of the variables in this problem.
       */
      const ComponentIndices component_indices;

      /**
       * The number of vector blocks. This equals $3+n_c$ where, in comparison
       * to the n_components field, the velocity components form a single
       * block.
       */
      const unsigned int n_blocks;

      /**
       * A structure that enumerates the vector blocks of the finite element
       * that correspond to each of the variables in this problem.
       */
      struct BlockIndices
      {
        unsigned int       velocities;
        unsigned int       pressure;
        //unsigned int       fluid_velocities;
        //unsigned int       fluid_pressure;
        //unsigned int       compaction_pressure;
        unsigned int       temperature;
        std::vector<unsigned int> compositional_fields;
      };
      /**
       * A variable that enumerates the vector blocks of the finite element
       * that correspond to each of the variables in this problem.
       */
      const BlockIndices block_indices;

      /**
       * A structure that contains FEValues extractors for every block of the
       * finite element used in the overall description.
       */
      struct Extractors
      {
        Extractors (const ComponentIndices &component_indices);

        const FEValuesExtractors::Vector              velocities;
        const FEValuesExtractors::Scalar              pressure;
        //const FEValuesExtractors::Vector              fluid_velocities;
        //const FEValuesExtractors::Scalar              fluid_pressure;
        //const FEValuesExtractors::Scalar              compaction_pressure;
        const FEValuesExtractors::Scalar              temperature;
        const std::vector<FEValuesExtractors::Scalar> compositional_fields;
      };
      /**
       * A variable that contains extractors for every block of the finite
       * element used in the overall description.
       */
      const Extractors extractors;

      /**
       * A structure that enumerates the base elements of the finite element
       * that correspond to each of the variables in this problem.
       *
       * If there are compositional fields, they are all discretized with the
       * same base element and, consequently, we only need a single index. If
       * a variable does not exist in the problem (e.g., we do not have
       * compositional fields), then the corresponding index is set to an
       * invalid number.
       */
      struct BaseElements
      {
        unsigned int       velocities;
        unsigned int       pressure;
        //unsigned int       fluid_velocities;
        //unsigned int       fluid_pressure;
        //unsigned int       compaction_pressure;
        unsigned int       temperature;
        unsigned int       compositional_fields;
      };
      /**
       * A variable that enumerates the base elements of the finite element
       * that correspond to each of the variables in this problem.
       */
      BaseElements base_elements;


      /**
       * A structure that contains component masks for each of the variables
       * in this problem. Component masks are a deal.II concept, see the
       * deal.II glossary.
       */
      struct ComponentMasks
      {
        ComponentMask              velocities;
        ComponentMask              pressure;
        //ComponentMask              fluid_velocities;
        //ComponentMask              fluid_pressure;
        //ComponentMask              compaction_pressure;
        ComponentMask              temperature;
        std::vector<ComponentMask> compositional_fields;
      };
      /**
       * A variable that contains component masks for each of the variables in
       * this problem. Component masks are a deal.II concept, see the deal.II
       * glossary.
       */
      ComponentMasks component_masks;

      /**
       * @}
       */

      /**
       * @name Things that depend on the current mesh
       * @{
       */
      /**
       * A variable that describes how many of the degrees of freedom on the
       * current mesh belong to each of the n_blocks blocks of the finite
       * element.
       */
      std::vector<types::global_dof_index> system_dofs_per_block;

      /**
       * A structure that contains index sets describing which of the globally
       * enumerated degrees of freedom are owned by or are relevant to the
       * current processor in a parallel computation.
       */
      struct IndexSets
      {
        /**
         * An index set that indicates which (among all) degrees of freedom
         * are relevant to the current processor. See the deal.II
         * documentation for the definition of the term "locally relevant
         * degrees of freedom".
         */
        IndexSet system_relevant_set;

        /**
         * A collection of index sets that for each of the vector blocks of
         * this finite element represents the global indices of the degrees of
         * freedom owned by this processor. The n_blocks elements of this
         * array form a mutually exclusive decomposition of the index set
         * containing all locally owned degrees of freedom.
         */
        std::vector<IndexSet> system_partitioning;

        /**
         * A collection of index sets that for each of the vector blocks of
         * this finite element represents the global indices of the degrees of
         * freedom are relevant to this processor. The n_blocks elements of
         * this array form a mutually exclusive decomposition of the index set
         * containing all locally relevant degrees of freedom, i.e., of the
         * system_relevant_set index set.
         */
        std::vector<IndexSet> system_relevant_partitioning;

        /**
         * A collection of index sets for each vector block of the Stokes
         * system (velocity and pressure). This variable contains the first
         * two elements of system_partitioning.
         */
        std::vector<IndexSet> stokes_partitioning;

        /**
         * Pressure unknowns that are locally owned. This IndexSet is needed
         * if velocity and pressure end up in the same block. If melt transport
         * is enabled, this will contain both pressures.
         */
        IndexSet locally_owned_pressure_dofs;

        /**
         * Fluid and compaction pressure unknowns that are locally owned. Only valid if
         * melt transport is enabled.
         */
        IndexSet locally_owned_melt_pressure_dofs;

        /**
         * Fluid pressure unknowns that are locally owned. Only valid if melt transport is enabled.
         */
        IndexSet locally_owned_fluid_pressure_dofs;
      };
      /**
       * A variable that contains index sets describing which of the globally
       * enumerated degrees of freedom are owned by the current processor in a
       * parallel computation.
       */
      IndexSets index_sets;

      /**
       * @}
       */

      /**
       * A function that gets the name of a compositional field as an input
       * parameter and returns its index. If the name is not found, an
       * exception is thrown.
       *
       * @param name The name of compositional field (as specified in the
       * input file)
       */
      unsigned int
      compositional_index_for_name (const std::string &name) const;

      /**
       * A function that gets the index of a compositional field as an input
       * parameter and returns its name.
       *
       * @param index The index of compositional field
       */
      std::string
      name_for_compositional_index (const unsigned int index) const;

      /**
       * A function that gets the name of a compositional field as an input
       * parameter and returns if the compositional field is used in this
       * simulation.
       *
       * @param name The name of compositional field (as specified in the
       * input file)
       */
      bool
      compositional_name_exists (const std::string &name) const;

    private:
      /**
       * A vector that stores the names of the compositional fields that will
       * be used in the simulation.
       */
      std::vector<std::string> composition_names;

  };
}


#endif
