// ---------------------------------------------------------------------
//
// Copyright (C) 2012 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------



// this function tests the correctness of the implementation of
// AffineConstraints<double>::distribute_local_to_global for ChunkSparseMatrix
// by comparing the results with a sparse matrix. As a test case, we use a
// square mesh that is refined once globally and then the first cell is refined
// adaptively.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/chunk_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <complex>
#include <iostream>

#include "../tests.h"

template <int dim>
void
test(unsigned int chunk_size)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.begin()->face(0)->set_boundary_id(1);
  tria.refine_global(1);
  tria.begin_active()->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof, constraints);
  VectorTools::interpolate_boundary_values(dof,
                                           1,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  SparsityPattern      sparsity;
  ChunkSparsityPattern chunk_sparsity;
  {
    DynamicSparsityPattern csp(dof.n_dofs(), dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, csp, constraints, false);
    sparsity.copy_from(csp);
    chunk_sparsity.copy_from(csp, chunk_size);
  }
  SparseMatrix<double>      sparse(sparsity);
  ChunkSparseMatrix<double> chunk_sparse(chunk_sparsity);

  FullMatrix<double> local_mat(fe.dofs_per_cell, fe.dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

  // loop over cells, fill local matrix with
  // random values, insert both into sparse and
  // full matrix. Make some random entries equal
  // to zero
  typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active(),
                                                 endc = dof.end();
  unsigned int counter                                = 0;
  for (; cell != endc; ++cell)
    {
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < fe.dofs_per_cell; ++j, ++counter)
          if (counter % 42 == 0)
            local_mat(i, j) = 0;
          else
            local_mat(i, j) = random_value<double>();
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_mat,
                                             local_dof_indices,
                                             sparse);
      constraints.distribute_local_to_global(local_mat,
                                             local_dof_indices,
                                             chunk_sparse);
    }

  // now check that the entries are indeed the
  // same
  double frobenius = 0.;
  for (unsigned int i = 0; i < sparse.m(); ++i)
    for (unsigned int j = 0; j < sparse.n(); ++j)
      frobenius += numbers::NumberTraits<double>::abs_square(
        sparse.el(i, j) - chunk_sparse.el(i, j));
  deallog << "Difference between chunk and sparse matrix: "
          << std::sqrt(frobenius) << std::endl;
}


int
main()
{
  initlog();
  deallog << std::setprecision(2);
  deallog.get_file_stream() << std::setprecision(2);

  test<2>(1);
  test<2>(2);
  test<2>(5);
}
