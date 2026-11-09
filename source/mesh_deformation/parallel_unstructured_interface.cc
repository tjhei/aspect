// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2014-2026 by the ASPECT authors.
//
// This file is part of ASPECT.
//
// Detailed license information governing the source code and
// contributions can be found in the folder LICENSES and in
// LICENSE and CONTRIBUTING.md at the top level directory.
//
// ------------------------------------------------------


#include <aspect/mesh_deformation/parallel_unstructured_interface.h>
#include <aspect/simulator_signals.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/linear_algebra_types.h>

#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools_evaluate.h>


namespace aspect
{
  namespace MeshDeformation
  {

    template <int dim>
    void
    ParallelUnstructuredInterface<dim>::
    compute_velocity_constraints_on_boundary(const DoFHandler<dim> &mesh_deformation_dof_handler,
                                             AffineConstraints<double> &mesh_velocity_constraints,
                                             const std::set<types::boundary_id> &boundary_ids) const
    {
      this->get_computing_timer().enter_subsection("Parallel Unstructured Interface: update velocity");

      // First compute a (global) vector that has the correct ASPECT solution at all boundary nodes.
      const std::vector<std::vector<double>> aspect_surface_solution = evaluate_aspect_solution_at_points();

      // Extract the velocity from the external tool. This is implemented in the derived class.
      const std::vector<Tensor<1,dim>> external_surface_velocities
        = compute_updated_velocities_at_points(aspect_surface_solution);

      // Take the velocities computed above and constrain the DoF's on the surface of ASPECT's mesh.
      const LinearAlgebra::Vector v_interpolated
        = interpolate_external_vector_field_to_surface_support_points(external_surface_velocities);

      // Create a ghosted vector that contains the interpolated velocities. This is necessary
      // because the constraints are set on all locally relevant DoFs (shared DoFs), not just
      // the locally owned ones.
      const DoFHandler<dim> &mesh_dof_handler = this->get_mesh_deformation_handler().get_mesh_deformation_dof_handler();
      const IndexSet mesh_locally_relevant = DoFTools::extract_locally_relevant_dofs (mesh_dof_handler);
      LinearAlgebra::Vector v_interpolated_ghosted(mesh_dof_handler.locally_owned_dofs(),
                                                   mesh_locally_relevant,
                                                   this->get_mpi_communicator());
      v_interpolated_ghosted = v_interpolated;

      // Set the constraints. For this, loop over all boundary DoFs
      // and if a boundary DoF is locally owned, create a constraint.
      // We later make that consistent across processors to ensure we
      // also know about the locally relevant DoFs constraints.
      // Now insert the relevant part of the solution into the mesh constraints
      const IndexSet constrained_dofs =
        DoFTools::extract_boundary_dofs(mesh_deformation_dof_handler,
                                        ComponentMask(dim, true),
                                        boundary_ids);

      for (const types::global_dof_index index : constrained_dofs)
        {
          if (mesh_velocity_constraints.can_store_line(index))
            if (mesh_velocity_constraints.is_constrained(index)==false)
              {
                mesh_velocity_constraints.add_constraint(index,
                                                         {},
                                                         v_interpolated_ghosted(index));
              }
        }
      this->get_computing_timer().leave_subsection("Parallel Unstructured Interface: update velocity");
    }


    template <int dim>
    void
    ParallelUnstructuredInterface<dim>::
    set_evaluation_points (const std::vector<Point<dim>> &points)
    {
      initialize_evaluation_points(points, {}, false);
    }


    template <int dim>
    void
    ParallelUnstructuredInterface<dim>::
    set_evaluation_points_with_weights (const std::vector<Point<dim>> &points,
                                        const std::vector<double> &weights)
    {
      initialize_evaluation_points(points, weights, true);
    }


    template <int dim>
    void
    ParallelUnstructuredInterface<dim>::
    initialize_evaluation_points (const std::vector<Point<dim>> &evaluation_points,
                                  const std::vector<double> &dual_weights,
                                  const bool weighted)
    {
      this->get_computing_timer().enter_subsection("Parallel Unstructured Interface: setup communication");

      // First, save a copy of the points at which we need the solution,
      // among other reasons so that we can track that input arguments
      // for later function calls describe the same number of points.
      const auto comm = this->get_mpi_communicator();
      const unsigned int weighted_mode = weighted;
      AssertThrow(Utilities::MPI::min(weighted_mode, comm) == Utilities::MPI::max(weighted_mode, comm),
                  ExcMessage("All ranks must select the same surface transfer method."));
      bool valid = !weighted || dual_weights.size() == evaluation_points.size();
      if (weighted)
        for (const double weight : dual_weights)
          valid = valid && std::isfinite(weight) && weight >= 0.0;
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), comm),
                  ExcMessage("Dual weights must match the points and be finite and nonnegative."));
      AssertThrow(!weighted || !this->get_geometry_model().has_curved_elements(),
                  ExcMessage("Weighted transfer on curved reference meshes has not yet been validated."));
      this->evaluation_points = evaluation_points;
      this->dual_weights = dual_weights;
      this->weighted_transfer = weighted;

      // Set up RemotePointEvaluation. The evaluation points are given in reference coordinates,
      // so we need to use a simple mapping instead of the MappingQEulerian stored in the Simulator. The latter
      // would produce the deformed mesh. We pick a high order mapping for curved meshes (sphere, shell),
      // in order to find evaluation points on the surface of the sphere/shell. For cartesian meshes, we
      // can use a Q1 mapping.
      static MappingQ<dim> mapping(this->get_geometry_model().has_curved_elements() ? 4 : 1);
      typename Utilities::MPI::RemotePointEvaluation<dim>::AdditionalData rpe_data;
      rpe_data.enforce_unique_mapping = weighted;
      remote_point_evaluator = std::make_unique<Utilities::MPI::RemotePointEvaluation<dim, dim>>(rpe_data);
      remote_point_evaluator->reinit(this->evaluation_points, this->get_triangulation(), mapping);

      AssertThrow(!weighted || remote_point_evaluator->all_points_found(),
                  ExcMessage("Every weighted surface point must be found in the reference mesh."));

#ifdef DEBUG
      if (!remote_point_evaluator->all_points_found())
        {

          this->get_pcout() << "WARNING: not all evaluation points were found inside the domain!" << std::endl;
          this->get_pcout() << "Evaluation points not found:" << std::endl;
          for (unsigned int p=0; p<evaluation_points.size(); ++p)
            {
              if (!remote_point_evaluator->point_found(p))
                {
                  this->get_pcout() << "Point " << p << ": " << evaluation_points[p] << std::endl;
                }
            }
        }
#endif

      // Create a mapping from evaluation points to support points. Note that one evaluation point can map to
      // multiple support points.
      if (!weighted_transfer)
        {
          // Deciding which evaluation point is closest to a support point requires somewhat complex MPI communication.
          // For now, we just gather all data on rank 0, do the computation, and then scatter the results back.
          const unsigned int n_mpi_processes = Utilities::MPI::n_mpi_processes(this->get_mpi_communicator());
          const unsigned int my_rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());

          const DoFHandler<dim> &mesh_dof_handler = this->get_mesh_deformation_handler().get_mesh_deformation_dof_handler();

          std::vector<double> squared_distances(mesh_dof_handler.locally_owned_dofs().size(), std::numeric_limits<double>::max());
          const DofToEvalPointData invalid
          {
            numbers::invalid_dof_index, numbers::invalid_unsigned_int, numbers::invalid_unsigned_int, numbers::invalid_unsigned_int, -1.0
          };
          std::vector<DofToEvalPointData> closest_evaluation_point_and_component(mesh_dof_handler.locally_owned_dofs().size(), invalid);

          // TODO: To support multiple boundaries with mesh deformation, we would need to know which boundary corresponds to a mesh
          // deformation model. For now, we just assume that there is only one boundary with mesh deformation.
          const auto boundary_ids = this->get_mesh_deformation_boundary_indicators();
          Assert(boundary_ids.size() == 1,
                 ExcMessage("Currently, we only support a single mesh deformation boundary."));

          const IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(mesh_dof_handler, ComponentMask(dim, true), boundary_ids);

          const unsigned int dofs_per_cell = mesh_dof_handler.get_fe().dofs_per_cell;
          std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

          // The remote_point_evaluator will gives us the velocities in all evaluation points that are within one of our locally
          // owned cells. The lambda defined below receives a list of points and their velocities for each cell. The coordinates
          // are given in coordinates of the unit cell.
          // For each support point of the velocity DoFHandler, we will try to find the closest evaluation point. We
          // do this by keeping track of the squared distance of the closest evaluation point checked so far.

          // For each evaluation point, we store the index and MPI rank to be able to identify them later:
          const unsigned int n_components = 2;
          std::vector<unsigned int> indices (evaluation_points.size() * n_components);
          for (unsigned int i=0; i<evaluation_points.size(); ++i)
            {
              indices[n_components*i] = i;
              indices[n_components*i+1] = my_rank;
            }

          // Note: We assume that process_and_evaluate() does not call our lambda concurrently, otherwise we would have write
          // conflicts when updating closest_evaluation_point_and_component and squared_distances.
          const auto eval_func = [&](const ArrayView<const unsigned int> &values,
                                     const typename Utilities::MPI::RemotePointEvaluation<dim>::CellData &cell_data)
          {
            std::vector<types::global_dof_index> cell_dof_indices (dofs_per_cell);
            for (const auto cell_index : cell_data.cell_indices())
              {
                const auto cell_dofs =
                  cell_data.get_active_cell_iterator(cell_index)->as_dof_handler_iterator(
                    mesh_dof_handler);
                cell_dofs->get_dof_indices(cell_dof_indices);

                const ArrayView<const Point<dim>> unit_points = cell_data.get_unit_points(cell_index);


                // Grab the values for this cell containing index and rank for each evaluation point.
                // Note: cell_data.get_data_view() does not work correctly with 2 components.
                const ArrayView<const unsigned int> local_values(
                  values.data() +
                  n_components*cell_data.reference_point_ptrs[cell_index],
                  n_components*(cell_data.reference_point_ptrs[cell_index + 1] -
                                cell_data.reference_point_ptrs[cell_index]));

                const std::vector< Point< dim >> &support_points = mesh_dof_handler.get_fe().get_unit_support_points();
                for (unsigned int j=0; j<support_points.size(); ++j)
                  {
                    // skip all DoFs in the interior
                    const bool is_boundary_dof = boundary_dofs.is_element(cell_dof_indices[j]);
                    if (!is_boundary_dof)
                      continue;

                    for (unsigned int i=0; i<unit_points.size(); ++i)
                      {
                        const unsigned int point_index = local_values[n_components*i];
                        const unsigned int rank = local_values[n_components*i+1];

                        const double distance_sq = unit_points[i].distance_square(support_points[j]);
                        if (distance_sq < squared_distances[cell_dof_indices[j]])
                          {
                            squared_distances[cell_dof_indices[j]] = distance_sq;
                            const unsigned int component = mesh_dof_handler.get_fe().system_to_component_index(j).first;
                            closest_evaluation_point_and_component[cell_dof_indices[j]] =
                              DofToEvalPointData {cell_dof_indices[j], rank, point_index, component, distance_sq};
                          }
                      }
                  }
              }
          };


          this->remote_point_evaluator->template process_and_evaluate<unsigned int, n_components>(indices, eval_func, /*sort_data*/ true);

          // remove DoFs not found (for example not surface DoFs):
          const auto new_end_it = std::remove_if(closest_evaluation_point_and_component.begin(),
                                                 closest_evaluation_point_and_component.end(),
                                                 [](const DofToEvalPointData &data)
          {
            return data.dof_index == numbers::invalid_dof_index;
          });
          closest_evaluation_point_and_component.erase(new_end_it, closest_evaluation_point_and_component.end());

          // send to rank 0 and find the closest evaluation point across all ranks, then scatter the results back to all ranks:
          std::vector<std::vector<DofToEvalPointData>> all_closest_evaluation_point_and_component
            = Utilities::MPI::gather(this->get_mpi_communicator(), closest_evaluation_point_and_component, /* root = */ 0);

          if (my_rank == 0)
            {
              // Combine data coming from all ranks and determine closest evaluation point for each DoF.
              // TODO: This might not be optimally scalable, for now we send everything to rank 0 and let
              // rank 0 find the closest evaluation point for each DoF, which is then scattered back to all
              // ranks.
              std::map<types::global_dof_index, DofToEvalPointData> map_from_dof;
              for (const auto &data : all_closest_evaluation_point_and_component)
                for (const auto &p : data)
                  {
                    const bool not_found = (map_from_dof.find(p.dof_index) == map_from_dof.end());
                    if (not_found || p.squared_distance < map_from_dof[p.dof_index].squared_distance)
                      map_from_dof[p.dof_index] = p;
                  }

              // Compile data for each rank and send it:
              std::vector<std::vector<DofToEvalPointData>> map_from_rank(n_mpi_processes);
              for (const auto &[dof_index, eval_point_data] : map_from_dof)
                map_from_rank[eval_point_data.evaluation_point_rank].push_back(eval_point_data);

              map_dof_to_eval_point = Utilities::MPI::scatter (this->get_mpi_communicator(), map_from_rank, 0);
            }
          else
            {
              // Receive my data from rank 0:
              std::vector<std::vector<DofToEvalPointData>> dummy;
              map_dof_to_eval_point = Utilities::MPI::scatter (this->get_mpi_communicator(), dummy, 0);
            }
        }

      // Finally, also ensure that upon mesh refinement, all of the
      // information set herein is invalidated:
      this->get_signals().pre_refinement_store_user_data
      .connect([this](typename parallel::distributed::Triangulation<dim> &)
      {
        this->evaluation_points.clear();
        this->dual_weights.clear();
        this->weighted_transfer = false;
        this->map_dof_to_eval_point.clear();
        this->remote_point_evaluator.reset();
      });
      this->get_computing_timer().leave_subsection("Parallel Unstructured Interface: setup communication");
    }



    template <int dim>
    std::vector<std::vector<double>>
    ParallelUnstructuredInterface<dim>::
    evaluate_aspect_solution_at_points () const
    {
      Assert (remote_point_evaluator != nullptr,
              ExcMessage("You can only call this function if you have previously "
                         "set the evaluation points by calling set_evaluation_points(), "
                         "and if the evaluator has not been invalidated by a mesh "
                         "refinement step."));

      // All components are evaluated (velocity, pressure, temperature, and N compositional fields).
      const unsigned int n_components = this->introspection().n_components;

      // Initialize the solution to be quiet_NaN() so that we know which points were not found
      // within the ASPECT domain.
      std::vector<std::vector<double>> solution_at_points (evaluation_points.size(), std::vector<double>(n_components, std::numeric_limits<double>::quiet_NaN()));

      // VectorTools::point_values can evaluate N components at a time, but this is a template argument and not a
      // runtime argument. For now, we just evaluate them one component at a time. Of course it would be more
      // efficient to branch and evaluate up to K at a time (for a reasonable number of K, say 10). Maybe something
      // to implement directly into deal.II.
      for (unsigned int c=0; c<n_components; ++c)
        {
          const std::vector<double> values = VectorTools::point_values<1>(*this->remote_point_evaluator,
                                                                          this->get_dof_handler(),
                                                                          this->get_solution(),
                                                                          dealii::VectorTools::EvaluationFlags::avg,
                                                                          c);
          for (unsigned int i=0; i<evaluation_points.size(); ++i)
            solution_at_points[i][c] = values[i];
        }

      return solution_at_points;
    }



    template <int dim>
    LinearAlgebra::Vector
    ParallelUnstructuredInterface<dim>::
    project_external_vector_field (const std::vector<Tensor<1,dim>> &values) const
    {
      const auto &dofs = this->get_mesh_deformation_handler().get_mesh_deformation_dof_handler();
      const auto comm = this->get_mpi_communicator();
      const auto relevant = DoFTools::extract_locally_relevant_dofs(dofs);
      const auto boundary_ids = this->get_mesh_deformation_boundary_indicators();
      const auto boundary = DoFTools::extract_boundary_dofs(dofs, ComponentMask(dim, true), boundary_ids);

      // Give each rank a contiguous range in the reduced boundary system.
      // Exchange numbering only for ghost boundary DoFs, rather than
      // replicating the global boundary numbering on every rank.
      const IndexSet owned_boundary = boundary & dofs.locally_owned_dofs();
      const auto [shift, n] = Utilities::MPI::partial_and_total_sum(owned_boundary.n_elements(), comm);
      AssertThrow(n > 0, ExcMessage("The projection boundary has no degrees of freedom."));
      std::map<types::global_dof_index, types::global_dof_index> numbering;
      for (const auto index : owned_boundary)
        numbering.emplace(index, shift + owned_boundary.index_within_set(index));

      IndexSet ghost_boundary = boundary & relevant;
      ghost_boundary.subtract_set(owned_boundary);
      // Use the complete volume ownership partition to look up owners:
      // boundary indices alone leave gaps in the volume index space.
      const auto owners = Utilities::MPI::compute_index_owner(dofs.locally_owned_dofs(), ghost_boundary, comm);
      std::map<unsigned int, std::vector<types::global_dof_index>> requests;
      for (const auto index : ghost_boundary)
        requests[owners[ghost_boundary.index_within_set(index)]].push_back(index);
      const auto requesters = Utilities::MPI::some_to_some(comm, requests);
      std::map<unsigned int, std::vector<std::pair<types::global_dof_index, types::global_dof_index>>> numbering_to_send;
      for (const auto &[rank, indices] : requesters)
        for (const auto index : indices)
          numbering_to_send[rank].emplace_back(index, numbering.at(index));

      const auto received_numbering = Utilities::MPI::some_to_some(comm, numbering_to_send);
      for (const auto &[rank, entries] : received_numbering)
        for (const auto &entry : entries)
          numbering.insert(entry);
      AssertDimension(numbering.size(), owned_boundary.n_elements() + ghost_boundary.n_elements());

      IndexSet owned(n), locally_relevant(n);
      owned.add_range(shift, shift + owned_boundary.n_elements());
      for (const auto &[index, reduced] : numbering)
        locally_relevant.add_index(reduced);
      owned.compress();
      locally_relevant.compress();

      AffineConstraints<double> volume_constraints(dofs.locally_owned_dofs(), relevant);
      DoFTools::make_hanging_node_constraints(dofs, volume_constraints);
      this->get_geometry_model().make_periodicity_constraints(dofs, volume_constraints);
      volume_constraints.close();
      AffineConstraints<double> constraints(owned, locally_relevant);
      for (const auto &[index, reduced] : numbering)
        if (relevant.is_element(index) && volume_constraints.is_constrained(index))
          {
            constraints.add_line(reduced);
            for (const auto &entry : *volume_constraints.get_constraint_entries(index))
              {
                AssertThrow(numbering.count(entry.first),
                            ExcMessage("A boundary constraint depends on a degree of freedom outside the projection boundary."));
                constraints.add_entry(reduced, numbering.at(entry.first), entry.second);
              }
          }
      constraints.close();

      LinearAlgebra::DynamicSparsityPattern sparsity(locally_relevant);
      std::vector<types::global_dof_index> cell_indices(dofs.get_fe().n_dofs_per_cell());
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          {
            cell->get_dof_indices(cell_indices);
            std::vector<types::global_dof_index> indices;
            for (const auto index : cell_indices)
              if (numbering.count(index))
                indices.push_back(numbering.at(index));
            constraints.add_entries_local_to_global(indices, sparsity, false);
          }
      SparsityTools::distribute_sparsity_pattern(sparsity, owned, comm, locally_relevant);
      LinearAlgebra::SparseMatrix mass;
      mass.reinit(owned, sparsity, comm);
      LinearAlgebra::Vector rhs(owned, comm), solution(owned, comm);
      const MappingQ<dim> mapping(1);
      FEFaceValues<dim> face_values(mapping, dofs.get_fe(), QGauss<dim-1>(dofs.get_fe().degree+1),
                                    update_values | update_JxW_values);
      const FEValuesExtractors::Vector vector(0);
      for (const auto &cell : dofs.active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
          {
            cell->get_dof_indices(cell_indices);
            std::vector<unsigned int> local_indices;
            std::vector<types::global_dof_index> indices;
            for (unsigned int i=0; i<cell_indices.size(); ++i)
              if (numbering.count(cell_indices[i]))
                {
                  local_indices.push_back(i);
                  indices.push_back(numbering.at(cell_indices[i]));
                }
            FullMatrix<double> local_mass(indices.size(), indices.size());
            for (const unsigned int face : cell->face_indices())
              if (cell->face(face)->at_boundary() && boundary_ids.count(cell->face(face)->boundary_id()))
                {
                  face_values.reinit(cell, face);
                  for (unsigned int q=0; q<face_values.n_quadrature_points; ++q)
                    for (unsigned int i=0; i<indices.size(); ++i)
                      for (unsigned int j=0; j<indices.size(); ++j)
                        local_mass(i,j) += face_values[vector].value(local_indices[i],q) *
                                           face_values[vector].value(local_indices[j],q) * face_values.JxW(q);
                }
            constraints.distribute_local_to_global(local_mass, indices, mass);
          }
      mass.compress(VectorOperation::add);

      std::vector<Tensor<1,dim>> weighted(values);
      for (unsigned int q=0; q<values.size(); ++q)
        weighted[q] *= dual_weights[q];
      const auto assemble = [&](const ArrayView<const Tensor<1,dim>> &data,
                                const typename Utilities::MPI::RemotePointEvaluation<dim>::CellData &cells)
      {
        for (const auto c : cells.cell_indices())
          {
            const auto cell = cells.get_active_cell_iterator(c)->as_dof_handler_iterator(dofs);
            const auto points = cells.get_unit_points(c);
            const auto local_values = cells.get_data_view(c, data);
            FEValues<dim> fe_values(mapping, dofs.get_fe(),
                                    Quadrature<dim>(std::vector<Point<dim>>(points.begin(), points.end())), update_values);
            fe_values.reinit(cell);
            cell->get_dof_indices(cell_indices);
            std::vector<types::global_dof_index> indices;
            std::vector<unsigned int> local_indices;
            for (unsigned int i=0; i<cell_indices.size(); ++i)
              if (numbering.count(cell_indices[i]))
                {
                  indices.push_back(numbering.at(cell_indices[i]));
                  local_indices.push_back(i);
                }
            Vector<double> local_rhs(indices.size());
            for (unsigned int q=0; q<points.size(); ++q)
              {
                bool on_boundary = false;
                for (const unsigned int face : cell->face_indices())
                  if (cell->face(face)->at_boundary() && boundary_ids.count(cell->face(face)->boundary_id()))
                    on_boundary = on_boundary || std::abs(points[q][face/2] - double(face%2)) < 1e-8;
                AssertThrow(on_boundary, ExcMessage("A weighted point is not on the projection boundary."));
                for (unsigned int i=0; i<indices.size(); ++i)
                  local_rhs[i] += fe_values[vector].value(local_indices[i],q) * local_values[q];
              }
            constraints.distribute_local_to_global(local_rhs, indices, rhs);
          }
      };
      remote_point_evaluator->template process_and_evaluate<Tensor<1,dim>,1>(weighted, assemble, true);
      rhs.compress(VectorOperation::add);
      LinearAlgebra::PreconditionJacobi preconditioner;
      preconditioner.initialize(mass);
      SolverControl control(5*n, 1e-12*rhs.l2_norm());
      SolverCG<LinearAlgebra::Vector> solver(control);
      solver.solve(mass, solution, rhs, preconditioner);
      constraints.distribute(solution);
      LinearAlgebra::Vector result(dofs.locally_owned_dofs(), comm);
      for (const auto index : owned_boundary)
        result[index] = solution[numbering.at(index)];
      result.compress(VectorOperation::insert);
      return result;
    }


    template <int dim>
    LinearAlgebra::Vector
    ParallelUnstructuredInterface<dim>::
    interpolate_external_vector_field_to_surface_support_points (const std::vector<Tensor<1,dim>> &velocities) const
    {
      Assert (remote_point_evaluator != nullptr,
              ExcMessage("You can only call this function if you have previously "
                         "set the evaluation points by calling set_evaluation_points(), "
                         "and if the evaluator has not been invalidated by a mesh "
                         "refinement step."));
      AssertDimension(velocities.size(), evaluation_points.size());


      if (weighted_transfer)
        return project_external_vector_field(velocities);

      // Create the output vector.
      const DoFHandler<dim> &mesh_dof_handler = this->get_mesh_deformation_handler().get_mesh_deformation_dof_handler();
      LinearAlgebra::Vector vector_with_surface_velocities(mesh_dof_handler.locally_owned_dofs(),
                                                           this->get_mpi_communicator());

      // For each entry in the mapping, we take the velocity from the evaluation point and insert it into the corresponding
      // DoF in the output vector.
      for (const auto &entry : map_dof_to_eval_point)
        vector_with_surface_velocities[entry.dof_index] = velocities[entry.evaluation_point_index][entry.component];

      // Because locally relevant DoFs are not necessarily locally owned, .compress() is needed to ensure that the correct
      // processor is assigning the value to the correct DoF.
      vector_with_surface_velocities.compress(VectorOperation::insert);

      return vector_with_surface_velocities;
    }
  }



  namespace MeshDeformation
  {
#define INSTANTIATE(dim) \
  template class ParallelUnstructuredInterface<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
