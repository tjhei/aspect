/*
  Copyright (C) 2018 - 2019 by the authors of the ASPECT code.

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


#ifndef _aspect_stokes_matrix_free_h
#define _aspect_stokes_matrix_free_h

#include <aspect/simulator.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/read_write_vector.h>

namespace aspect
{
  using namespace dealii;


  namespace MatrixFreeStokesOperators
  {

    template <int dim, int degree, typename number>
    class StokesOperator
      : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >
    {
      public:
        StokesOperator ();
        void clear ();
        void fill_viscosities_and_pressure_scaling(const Table<2, VectorizedArray<number> > &visc_table,
                                                   const double scaling);
        Table<2, VectorizedArray<number> > get_visc_table();
        virtual void compute_diagonal ();

      private:
        virtual void apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                                const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const;

        void local_apply (const dealii::MatrixFree<dim, number> &data,
                          dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                          const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
                          const std::pair<unsigned int, unsigned int> &cell_range) const;

        Table<2, VectorizedArray<number> > viscosity_x_2;
        double pressure_scaling;
    };
    template <int dim, int degree, typename number>
    StokesOperator<dim,degree,number>::StokesOperator ()
      :
      MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >()
    {}
    template <int dim, int degree, typename number>
    void
    StokesOperator<dim,degree,number>::clear ()
    {
      viscosity_x_2.reinit(0, 0);
      MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::BlockVector<number> >::clear();
    }
    template <int dim, int degree, typename number>
    void
    StokesOperator<dim,degree,number>::
    fill_viscosities_and_pressure_scaling (const Table<2, VectorizedArray<number> > &visc_table,
                                           const double scaling)
    {
      FEEvaluation<dim,degree,degree+1,dim,number> velocity (*this->data, 0);
      Assert(visc_table.n_elements() == this->data->n_macro_cells()*velocity.n_q_points, ExcMessage("Tables are not the right size!"));

      viscosity_x_2 = visc_table;
      pressure_scaling = scaling;
    }
    template <int dim, int degree, typename number>
    Table<2, VectorizedArray<number> >
    StokesOperator<dim,degree,number>::get_visc_table()
    {
      return viscosity_x_2;
    }

    template <int dim, int degree, typename number>
    void
    StokesOperator<dim,degree,number>
    ::local_apply (const dealii::MatrixFree<dim, number>                 &data,
                   dealii::LinearAlgebra::distributed::BlockVector<number>       &dst,
                   const dealii::LinearAlgebra::distributed::BlockVector<number> &src,
                   const std::pair<unsigned int, unsigned int>           &cell_range) const
    {
      typedef VectorizedArray<number> vector_t;
      FEEvaluation<dim,degree,degree+1,dim,number> velocity (data, 0);
      FEEvaluation<dim,degree-1,  degree+1,1,  number> pressure (data, 1);

      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          velocity.reinit (cell);
          velocity.read_dof_values (src.block(0));
          velocity.evaluate (false,true,false);
          pressure.reinit (cell);
          pressure.read_dof_values (src.block(1));
          pressure.evaluate (true,false,false);

          for (unsigned int q=0; q<velocity.n_q_points; ++q)
            {
              SymmetricTensor<2,dim,vector_t> sym_grad_u =
                velocity.get_symmetric_gradient (q);
              vector_t pres = pressure.get_value(q);
              vector_t div = -trace(sym_grad_u);
              pressure.submit_value   (pressure_scaling*div, q);

              sym_grad_u *= viscosity_x_2(cell,q);
              // subtract p * I
              for (unsigned int d=0; d<dim; ++d)
                sym_grad_u[d][d] -= pressure_scaling*pres;

              velocity.submit_symmetric_gradient(sym_grad_u, q);
            }

          velocity.integrate (false,true);
          velocity.distribute_local_to_global (dst.block(0));
          pressure.integrate (true,false);
          pressure.distribute_local_to_global (dst.block(1));
        }
    }
    template <int dim, int degree, typename number>
    void
    StokesOperator<dim,degree,number>
    ::apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                 const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const
    {
      MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >::
      data->cell_loop(&StokesOperator::local_apply, this, dst, src);
    }
    template <int dim, int degree, typename number>
    void
    StokesOperator<dim,degree,number>
    ::compute_diagonal ()
    {
      Assert(false, ExcNotImplemented());
    }


    template <int dim, int degree_p, typename number>
    class MassMatrixOperator
      : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>
    {
      public:
        MassMatrixOperator ();
        void clear ();
        void fill_viscosities_and_pressure_scaling(const Table<2, VectorizedArray<number> > &visc_table,
                                                   const double scaling);
        virtual void compute_diagonal ();

      private:
        virtual void apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
                                const dealii::LinearAlgebra::distributed::Vector<number> &src) const;

        void local_apply (const dealii::MatrixFree<dim, number> &data,
                          dealii::LinearAlgebra::distributed::Vector<number> &dst,
                          const dealii::LinearAlgebra::distributed::Vector<number> &src,
                          const std::pair<unsigned int, unsigned int> &cell_range) const;

        void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                                     dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                                     const unsigned int                               &dummy,
                                     const std::pair<unsigned int,unsigned int>       &cell_range) const;

        Table<2, VectorizedArray<number> > one_over_viscosity;
        double pressure_scaling;
    };
    template <int dim, int degree_p, typename number>
    MassMatrixOperator<dim,degree_p,number>::MassMatrixOperator ()
      :
      MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number> >()
    {}
    template <int dim, int degree_p, typename number>
    void
    MassMatrixOperator<dim,degree_p,number>::clear ()
    {
      one_over_viscosity.reinit(0, 0);
      MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::clear();
    }
    template <int dim, int degree_p, typename number>
    void
    MassMatrixOperator<dim,degree_p,number>::
    fill_viscosities_and_pressure_scaling (const Table<2, VectorizedArray<number> > &visc_table,
                                           const double scaling)
    {
      pressure_scaling = scaling;

      FEEvaluation<dim,degree_p,degree_p+2,dim,number> velocity (*this->data, 0);
      Assert(visc_table.n_elements() == this->data->n_macro_cells()*velocity.n_q_points, ExcMessage("Tables are not the right size!"));

      one_over_viscosity = visc_table;
    }
    template <int dim, int degree_p, typename number>
    void
    MassMatrixOperator<dim,degree_p,number>
    ::local_apply (const dealii::MatrixFree<dim, number>                 &data,
                   dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                   const dealii::LinearAlgebra::distributed::Vector<number> &src,
                   const std::pair<unsigned int, unsigned int>           &cell_range) const
    {
      FEEvaluation<dim,degree_p,degree_p+2,1,number> pressure (data);

      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          AssertDimension(one_over_viscosity.size(0), data.n_macro_cells());
          AssertDimension(one_over_viscosity.size(1), pressure.n_q_points);

          pressure.reinit (cell);
          pressure.read_dof_values(src);
          pressure.evaluate (true, false);
          for (unsigned int q=0; q<pressure.n_q_points; ++q)
            pressure.submit_value(one_over_viscosity(cell,q)*pressure_scaling*pressure_scaling*
                                  pressure.get_value(q),q);
          pressure.integrate (true, false);
          pressure.distribute_local_to_global (dst);
        }
    }
    template <int dim, int degree_p, typename number>
    void
    MassMatrixOperator<dim,degree_p,number>
    ::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
                 const dealii::LinearAlgebra::distributed::Vector<number> &src) const
    {
      MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::
      data->cell_loop(&MassMatrixOperator::local_apply, this, dst, src);
    }
    template <int dim, int degree_p, typename number>
    void
    MassMatrixOperator<dim,degree_p,number>
    ::compute_diagonal ()
    {
      this->inverse_diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());
      this->diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());

      dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
        this->inverse_diagonal_entries->get_vector();
      dealii::LinearAlgebra::distributed::Vector<number> &diagonal =
        this->diagonal_entries->get_vector();

      unsigned int dummy = 0;
      this->data->initialize_dof_vector(inverse_diagonal);
      this->data->initialize_dof_vector(diagonal);

      this->data->cell_loop (&MassMatrixOperator::local_compute_diagonal, this,
                             diagonal, dummy);

      this->set_constrained_entries_to_one(diagonal);
      inverse_diagonal = diagonal;
      const unsigned int local_size = inverse_diagonal.local_size();
      for (unsigned int i=0; i<local_size; ++i)
        {
          Assert(inverse_diagonal.local_element(i) > 0.,
                 ExcMessage("No diagonal entry in a positive definite operator "
                            "should be zero"));
          inverse_diagonal.local_element(i)
            =1./inverse_diagonal.local_element(i);
        }
    }
    template <int dim, int degree_p, typename number>
    void
    MassMatrixOperator<dim,degree_p,number>
    ::local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                              dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                              const unsigned int &,
                              const std::pair<unsigned int,unsigned int>       &cell_range) const
    {
      FEEvaluation<dim,degree_p,degree_p+2,1,number> pressure (data, 0);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          pressure.reinit (cell);
          AlignedVector<VectorizedArray<number> > diagonal(pressure.dofs_per_cell);
          for (unsigned int i=0; i<pressure.dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<pressure.dofs_per_cell; ++j)
                pressure.begin_dof_values()[j] = VectorizedArray<number>();
              pressure.begin_dof_values()[i] = make_vectorized_array<number> (1.);

              pressure.evaluate (true,false,false);
              for (unsigned int q=0; q<pressure.n_q_points; ++q)
                {
                  //std::cout << one_over_viscosity(cell,q)[0] << std::endl;
                  pressure.submit_value(one_over_viscosity(cell,q)*pressure_scaling*pressure_scaling*
                                        pressure.get_value(q),q);
                }
              pressure.integrate (true,false);

              diagonal[i] = pressure.begin_dof_values()[i];
            }

          for (unsigned int i=0; i<pressure.dofs_per_cell; ++i)
            pressure.begin_dof_values()[i] = diagonal[i];
          pressure.distribute_local_to_global (dst);
        }
    }



    template <int dim, int degree, typename number>
    class ABlockOperator
      : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>
    {
      public:
        ABlockOperator ();
        void clear ();
        void fill_viscosities(const Table<2, VectorizedArray<number> > &visc_table);
        virtual void compute_diagonal ();

      private:
        virtual void apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
                                const dealii::LinearAlgebra::distributed::Vector<number> &src) const;

        void local_apply (const dealii::MatrixFree<dim, number> &data,
                          dealii::LinearAlgebra::distributed::Vector<number> &dst,
                          const dealii::LinearAlgebra::distributed::Vector<number> &src,
                          const std::pair<unsigned int, unsigned int> &cell_range) const;

        void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                                     dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                                     const unsigned int                               &dummy,
                                     const std::pair<unsigned int,unsigned int>       &cell_range) const;

        Table<2, VectorizedArray<number> > viscosity_x_2;
    };
    template <int dim, int degree, typename number>
    ABlockOperator<dim,degree,number>::ABlockOperator ()
      :
      MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<number> >()
    {}
    template <int dim, int degree, typename number>
    void
    ABlockOperator<dim,degree,number>::clear ()
    {
      viscosity_x_2.reinit(0, 0);
      MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::clear();
    }
    template <int dim, int degree, typename number>
    void
    ABlockOperator<dim,degree,number>::
    fill_viscosities (const Table<2, VectorizedArray<number> > &visc_table)
    {
      FEEvaluation<dim,degree,degree+1,dim,number> velocity (*this->data, 0);
      Assert(visc_table.n_elements() == this->data->n_macro_cells()*velocity.n_q_points, ExcMessage("Tables are not the right size!"));

      viscosity_x_2 = visc_table;
    }
    template <int dim, int degree, typename number>
    void
    ABlockOperator<dim,degree,number>
    ::local_apply (const dealii::MatrixFree<dim, number>                 &data,
                   dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                   const dealii::LinearAlgebra::distributed::Vector<number> &src,
                   const std::pair<unsigned int, unsigned int>           &cell_range) const
    {
      FEEvaluation<dim,degree,degree+1,dim,number> velocity (data,0);

      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          AssertDimension(viscosity_x_2.size(0), data.n_macro_cells());
          AssertDimension(viscosity_x_2.size(1), velocity.n_q_points);

          velocity.reinit (cell);
          velocity.read_dof_values(src);
          velocity.evaluate (false, true, false);
          for (unsigned int q=0; q<velocity.n_q_points; ++q)
            {
              velocity.submit_symmetric_gradient
              (viscosity_x_2(cell,q)*velocity.get_symmetric_gradient(q),q);
            }
          velocity.integrate (false, true);
          velocity.distribute_local_to_global (dst);
        }
    }
    template <int dim, int degree, typename number>
    void
    ABlockOperator<dim,degree,number>
    ::apply_add (dealii::LinearAlgebra::distributed::Vector<number> &dst,
                 const dealii::LinearAlgebra::distributed::Vector<number> &src) const
    {
      MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<number> >::
      data->cell_loop(&ABlockOperator::local_apply, this, dst, src);
    }
    template <int dim, int degree, typename number>
    void
    ABlockOperator<dim,degree,number>
    ::compute_diagonal ()
    {
      this->inverse_diagonal_entries.
      reset(new DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number> >());
      dealii::LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
        this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal);
      unsigned int dummy = 0;
      this->data->cell_loop (&ABlockOperator::local_compute_diagonal, this,
                             inverse_diagonal, dummy);

      this->set_constrained_entries_to_one(inverse_diagonal);

      for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
        {
          Assert(inverse_diagonal.local_element(i) > 0.,
                 ExcMessage("No diagonal entry in a positive definite operator "
                            "should be zero"));
          inverse_diagonal.local_element(i) =
            1./inverse_diagonal.local_element(i);
        }
    }
    template <int dim, int degree, typename number>
    void
    ABlockOperator<dim,degree,number>
    ::local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                              dealii::LinearAlgebra::distributed::Vector<number>  &dst,
                              const unsigned int &,
                              const std::pair<unsigned int,unsigned int>       &cell_range) const
    {
      FEEvaluation<dim,degree,degree+1,dim,number> velocity (data, 0);
      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
        {
          velocity.reinit (cell);
          AlignedVector<VectorizedArray<number> > diagonal(velocity.dofs_per_cell);
          for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<velocity.dofs_per_cell; ++j)
                velocity.begin_dof_values()[j] = VectorizedArray<number>();
              velocity.begin_dof_values()[i] = make_vectorized_array<number> (1.);

              velocity.evaluate (false,true,false);
              for (unsigned int q=0; q<velocity.n_q_points; ++q)
                {
                  velocity.submit_symmetric_gradient
                  (viscosity_x_2(cell,q)*velocity.get_symmetric_gradient(q),q);
                }
              velocity.integrate (false,true);

              diagonal[i] = velocity.begin_dof_values()[i];
            }

          for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
            velocity.begin_dof_values()[i] = diagonal[i];
          velocity.distribute_local_to_global (dst);
        }
    }


// Class for computing the coefficient table. It's necessary since the parallel ordering
// of cells in matrix-free is different between DG and Continuous elements.
    template <int dim, int degree_q, typename number>
    class ComputeCoefficientProjection
      : public MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number>>
    {
      public:
        ComputeCoefficientProjection ();
        void clear ();

        Table<2, VectorizedArray<number> >
        return_viscosity_table(const dealii::LinearAlgebra::distributed::Vector<number> &dof_vals_dg);


        virtual void compute_diagonal ();
      private:
        virtual void apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                                const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const;
    };
    template <int dim, int degree_q, typename number>
    ComputeCoefficientProjection<dim,degree_q,number>::ComputeCoefficientProjection ()
      :
      MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::BlockVector<number> >()
    {}
    template <int dim, int degree_q, typename number>
    void
    ComputeCoefficientProjection<dim,degree_q,number>::clear ()
    {
      MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::BlockVector<number> >::clear();
    }

    template <int dim, int degree_q, typename number>
    Table<2, VectorizedArray<number> >
    ComputeCoefficientProjection<dim,degree_q,number>::
    return_viscosity_table (const dealii::LinearAlgebra::distributed::Vector<number> &dof_vals_dg)
    {
      const unsigned int n_cells = this->data->n_macro_cells();
      FEEvaluation<dim,0,  degree_q, 1,  number> projection (*this->data, 1);

      Table<2, VectorizedArray<number> > viscosity_table(n_cells, projection.n_q_points);
      for (unsigned int cell=0; cell<n_cells; ++cell)
        {
          projection.reinit (cell);
          projection.read_dof_values(dof_vals_dg);
          projection.evaluate(true,false,false);

          for (unsigned int q=0; q<projection.n_q_points; ++q)
            viscosity_table(cell,q) = projection.get_value(q);
        }

      return viscosity_table;
    }
    template <int dim, int degree_q, typename number>
    void
    ComputeCoefficientProjection<dim,degree_q,number>
    ::apply_add (dealii::LinearAlgebra::distributed::BlockVector<number> &dst,
                 const dealii::LinearAlgebra::distributed::BlockVector<number> &src) const
    {
      // Function not needed, but must be defined
      (void)dst;
      (void)src;
      Assert(false, ExcNotImplemented());
    }
    template <int dim, int degree_q, typename number>
    void
    ComputeCoefficientProjection<dim,degree_q,number>
    ::compute_diagonal ()
    {
      // Function not needed, but must be defined
      Assert(false, ExcNotImplemented());
    }
  }

  template<int dim>
  class StokesMatrixFreeHandler
  {
    public:
      /**
               * Initialize this class, allowing it to read in
               * relevant parameters as well as giving it a reference to the
               * Simulator that owns it, since it needs to make fairly extensive
               * changes to the internals of the simulator.
               */
      StokesMatrixFreeHandler(Simulator<dim> &, ParameterHandler &prm);

      /**
               * Destructor for the free surface handler.
               */
      ~StokesMatrixFreeHandler();

      /**
               * The main execution step
               */
      std::pair<double,double> solve();

      /**
               * Allocates and sets up the members of the FreeSurfaceHandler. This
               * is called by Simulator<dim>::setup_dofs()
               */
      void setup_dofs();

      /**
           * Evalute the MaterialModel to query for the viscosity on the active cells
           * and cache the information for later usage.
           */
      void evaluate_viscosity();

      /**
           * Add correction to system RHS
           */
      void correct_stokes_rhs();

      /**
               * Declare parameters.
               */
      static
      void declare_parameters (ParameterHandler &prm);

      /**
               * Parse parameters
               */
      void parse_parameters (ParameterHandler &prm);

    private:
      /**
               * Reference to the Simulator object to which a FreeSurfaceHandler
               * instance belongs.
               */
      Simulator<dim> &sim;

      // TODO: velocity degree not only 2, Choosing quadrature degree?
      typedef MatrixFreeStokesOperators::StokesOperator<dim,2,double> StokesMatrixType;
      typedef MatrixFreeStokesOperators::MassMatrixOperator<dim,1,double> MassMatrixType;
      typedef MatrixFreeStokesOperators::ABlockOperator<dim,2,double> ABlockMatrixType;


      FESystem<dim> stokes_fe;
      FESystem<dim> fe_v;
      FESystem<dim> fe_p;

      DoFHandler<dim> dof_handler_v;
      DoFHandler<dim> dof_handler_p;

      StokesMatrixType stokes_matrix;
      MassMatrixType mass_matrix;

      ConstraintMatrix stokes_constraints;
      ConstraintMatrix constraints_v;
      ConstraintMatrix constraints_p;

      MGLevelObject<ABlockMatrixType> mg_matrices;
      MGConstrainedDoFs              mg_constrained_dofs;

      // Stuff for coefficient projection
      DoFHandler<dim> dof_handler_projection;
      ConstraintMatrix constraints_projection;
      MGConstrainedDoFs mg_constrained_dofs_projection;
      const FESystem<dim> fe_projection;

      dealii::LinearAlgebra::distributed::Vector<double> active_coef_dof_vec;
      MGLevelObject<dealii::LinearAlgebra::distributed::Vector<double> > level_coef_dof_vec;


      MGTransferMatrixFree<dim,double> mg_transfer;

      //friend class Simulator<dim>;
      //friend class SimulatorAccess<dim>;
  };
}


#endif
