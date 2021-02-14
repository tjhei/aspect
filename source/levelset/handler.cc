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
#include <aspect/parameters.h>
#include <aspect/levelset/handler.h>

using namespace dealii;

namespace aspect
{
  namespace internal
  {
//    namespace Assembly
//    {
//      namespace Scratch
//      {
//        template <int dim>
//        VolumeOfFluidSystem<dim>::VolumeOfFluidSystem (const FiniteElement<dim> &finite_element,
//                                                       const FiniteElement<dim> &volume_of_fluid_element,
//                                                       const Mapping<dim>       &mapping,
//                                                       const Quadrature<dim>    &quadrature,
//                                                       const Quadrature<dim-1>  &face_quadrature)
//          :
//          finite_element_values (mapping,
//                                 finite_element, quadrature,
//                                 update_values |
//                                 update_gradients |
//                                 update_JxW_values),
//          neighbor_finite_element_values (mapping,
//                                          finite_element, quadrature,
//                                          update_values |
//                                          update_gradients |
//                                          update_JxW_values),
//          face_finite_element_values (mapping,
//                                      finite_element, face_quadrature,
//                                      update_values |
//                                      update_quadrature_points |
//                                      update_gradients |
//                                      update_normal_vectors |
//                                      update_JxW_values),
//          neighbor_face_finite_element_values (mapping,
//                                               finite_element, face_quadrature,
//                                               update_values |
//                                               update_gradients |
//                                               update_normal_vectors |
//                                               update_JxW_values),
//          subface_finite_element_values (mapping,
//                                         finite_element, face_quadrature,
//                                         update_values |
//                                         update_gradients |
//                                         update_normal_vectors |
//                                         update_JxW_values),
//          local_dof_indices(finite_element.dofs_per_cell),
//          phi_field (volume_of_fluid_element.dofs_per_cell, numbers::signaling_nan<double>()),
//          old_field_values (quadrature.size(), numbers::signaling_nan<double>()),
//          cell_i_n_values (quadrature.size(), numbers::signaling_nan<Tensor<1, dim> > ()),
//          cell_i_d_values (quadrature.size(), numbers::signaling_nan<double> ()),
//          face_current_velocity_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1, dim> >()),
//          face_old_velocity_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1, dim> >()),
//          face_old_old_velocity_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1, dim> >()),
//          neighbor_old_values (face_quadrature.size(), numbers::signaling_nan<double>()),
//          neighbor_i_n_values (face_quadrature.size(), numbers::signaling_nan<Tensor<1, dim> >()),
//          neighbor_i_d_values (face_quadrature.size(), numbers::signaling_nan<double>())
//        {}



//        template <int dim>
//        VolumeOfFluidSystem<dim>::VolumeOfFluidSystem (const VolumeOfFluidSystem &scratch)
//          :
//          finite_element_values (scratch.finite_element_values.get_mapping(),
//                                 scratch.finite_element_values.get_fe(),
//                                 scratch.finite_element_values.get_quadrature(),
//                                 scratch.finite_element_values.get_update_flags()),
//          neighbor_finite_element_values (scratch.neighbor_finite_element_values.get_mapping(),
//                                          scratch.neighbor_finite_element_values.get_fe(),
//                                          scratch.neighbor_finite_element_values.get_quadrature(),
//                                          scratch.neighbor_finite_element_values.get_update_flags()),
//          face_finite_element_values (scratch.face_finite_element_values.get_mapping(),
//                                      scratch.face_finite_element_values.get_fe(),
//                                      scratch.face_finite_element_values.get_quadrature(),
//                                      scratch.face_finite_element_values.get_update_flags()),
//          neighbor_face_finite_element_values (scratch.neighbor_face_finite_element_values.get_mapping(),
//                                               scratch.neighbor_face_finite_element_values.get_fe(),
//                                               scratch.neighbor_face_finite_element_values.get_quadrature(),
//                                               scratch.neighbor_face_finite_element_values.get_update_flags()),
//          subface_finite_element_values (scratch.subface_finite_element_values.get_mapping(),
//                                         scratch.subface_finite_element_values.get_fe(),
//                                         scratch.subface_finite_element_values.get_quadrature(),
//                                         scratch.subface_finite_element_values.get_update_flags()),
//          local_dof_indices (scratch.finite_element_values.get_fe().dofs_per_cell),
//          phi_field (scratch.phi_field),
//          old_field_values (scratch.old_field_values),
//          cell_i_n_values (scratch.cell_i_n_values),
//          cell_i_d_values (scratch.cell_i_d_values),
//          face_current_velocity_values (scratch.face_current_velocity_values),
//          face_old_velocity_values (scratch.face_old_velocity_values),
//          face_old_old_velocity_values (scratch.face_old_old_velocity_values),
//          neighbor_old_values (scratch.neighbor_old_values),
//          neighbor_i_n_values (scratch.neighbor_i_n_values),
//          neighbor_i_d_values (scratch.neighbor_i_d_values)
//        {}
//      }



//      namespace CopyData
//      {
//        template <int dim>
//        VolumeOfFluidSystem<dim>::VolumeOfFluidSystem(const FiniteElement<dim> &finite_element)
//          :
//          local_matrix (finite_element.dofs_per_cell,
//                        finite_element.dofs_per_cell),
//          local_rhs (finite_element.dofs_per_cell),
//          local_dof_indices (finite_element.dofs_per_cell)
//        {
//          TableIndices<2> mat_size(finite_element.dofs_per_cell,
//                                   finite_element.dofs_per_cell);
//          for (unsigned int i=0;
//               i < GeometryInfo<dim>::max_children_per_face *GeometryInfo<dim>::faces_per_cell;
//               ++i)
//            {
//              face_contributions_mask[i] = false;
//              local_face_rhs[i].reinit (finite_element.dofs_per_cell);
//              local_face_matrices_ext_ext[i].reinit(mat_size);
//              neighbor_dof_indices[i].resize(finite_element.dofs_per_cell);
//            }
//        }



//        template<int dim>
//        VolumeOfFluidSystem<dim>::VolumeOfFluidSystem(const VolumeOfFluidSystem &data)
//          :
//          local_matrix (data.local_matrix),
//          local_rhs (data.local_rhs),
//          local_face_rhs (data.local_face_rhs),
//          local_face_matrices_ext_ext (data.local_face_matrices_ext_ext),
//          local_dof_indices (data.local_dof_indices),
//          neighbor_dof_indices (data.neighbor_dof_indices)
//        {
//          for (unsigned int i=0;
//               i < GeometryInfo<dim>::max_children_per_face *GeometryInfo<dim>::faces_per_cell;
//               ++i)
//            {
//              face_contributions_mask[i] = false;
//            }
//        }
//      }
  }




//  template <int dim>
//  LevelsetHandler<dim>::LevelsetHandler(const FEVariable<dim> &volume_fraction,
//                                              const FEVariable<dim> &reconstruction,
//                                              const FEVariable<dim> &level_set,
//                                              const unsigned int composition_index)
//    : volume_fraction (volume_fraction),
//      reconstruction (reconstruction),
//      level_set (level_set),
//      composition_index(composition_index)
//  {}



  template <int dim>
  LevelsetHandler<dim>::LevelsetHandler (Simulator<dim> &simulator,
                                         ParameterHandler &prm)
    : sim (simulator)
  {
    this->initialize_simulator(sim);
//    assembler.initialize_simulator(sim);
    parse_parameters (prm);

    this->get_signals().post_set_initial_state.connect(
      [&](const SimulatorAccess<dim> &)
    {
      //this->set_initial_volume_fractions();
    });
  }






  template <int dim>
  void
  LevelsetHandler<dim>::declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("Levelset");

    prm.leave_subsection();

//    prm.enter_subsection ("Volume of Fluid");
//    {
//      prm.declare_entry ("Volume fraction threshold", "1e-6",
//                         Patterns::Double (0., 1.),
//                         "Minimum significant volume. Fluid fractions below this value are considered to be zero.");

//      prm.declare_entry ("Volume of Fluid solver tolerance", "1e-12",
//                         Patterns::Double(0., 1.),
//                         "The relative tolerance up to which the linear system "
//                         "for the Volume of Fluid system gets solved. See "
//                         "'Solver parameters/Composition solver tolerance' "
//                         "for more details.");

//      prm.declare_entry ("Number initialization samples", "3",
//                         Patterns::Integer (1),
//                         "Number of divisions per dimension when computing the initial volume fractions."
//                         "If set to the default of 3 for a 2D model, then initialization will be based on "
//                         "the initialization criterion at $3^2=9$ points within each cell. If the initialization "
//                         "based on a composition style initial condition, a larger value may be desired for better "
//                         "approximation of the initial fluid fractions. Smaller values will suffice in the case of "
//                         "level set initializations due to the presence of more information to better approximate "
//                         "the initial fluid fractions.");
//    }
//    prm.leave_subsection ();

//    prm.enter_subsection("Initial composition model");
//    {
//      prm.declare_entry("Volume of fluid initialization type", "",
//                        Patterns::Map (Patterns::Anything(),
//                                       Patterns::Selection("composition|level set")),
//                        "A comma separated list denoting the method to be used to "
//                        "initialize a composition field specified to be advected using "
//                        "the volume of fluid method.\n\n"
//                        "The format of valid entries for this parameter is that "
//                        "of a map given as ``key1:value1, key2:value2`` where "
//                        "each key must be the name of a compositional field "
//                        "using the volume of fluid advection method, and the "
//                        "value is one of ``composition`` or ``level "
//                        "set``. ``composition`` is the default\n\n"
//                        "When ``composition is specified, the initial model is "
//                        "treated as a standard composition field with bounds "
//                        "between 0 and 1 assumed, The initial fluid fractions "
//                        "are then based on an iterated midpoint quadrature. "
//                        "Resultant volume fractions outside of the bounds will be "
//                        "coerced to the nearest valid value (ie 0 or 1). "
//                        "If ``level set`` is specified, the intial data will be assumed to "
//                        "be in the form of a signed distance level set function "
//                        "(i.e. a function which is positive when in the "
//                        "fluid, negative outside, and zero on the interface "
//                        "and the magnitude is always the distance to the "
//                        "interface so the gradient is one everywhere).");
//    }
//    prm.leave_subsection();
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
//    AssertThrow(dim==2,
//                ExcMessage("Volume of Fluid Interface Tracking is currently only functional for dim=2."));

//    AssertThrow(this->get_parameters().CFL_number < 1.0,
//                ExcMessage("Volume of Fluid Interface Tracking requires CFL < 1."));

//    AssertThrow(!this->get_material_model().is_compressible(),
//                ExcMessage("Volume of Fluid Interface Tracking currently assumes incompressibility."));

//    AssertThrow(dynamic_cast<const MappingCartesian<dim> *>(&(this->get_mapping())),
//                ExcMessage("Volume of Fluid Interface Tracking currently requires Cartesian Mappings"));

//    AssertThrow(!this->get_parameters().mesh_deformation_enabled,
//                ExcMessage("Volume of Fluid Interface Tracking is currently incompatible with the Free Surface implementation."));

//    AssertThrow(!this->get_parameters().include_melt_transport,
//                ExcMessage("Volume of Fluid Interface Tracking has not been tested with melt transport yet, so inclusion of both is currently disabled."))

  }








}



namespace aspect
{
#define INSTANTIATE(dim) \
  template class LevelsetHandler<dim>;

  ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
}
