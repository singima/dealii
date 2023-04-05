// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2022 by the deal.II authors
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



#include <deal.II/base/config.h>

#  include <deal.II/base/utilities.h>

#  include <deal.II/lac/block_vector.h>
#  include <deal.II/lac/la_parallel_block_vector.h>
#  include <deal.II/lac/la_parallel_vector.h>
#  ifdef DEAL_II_WITH_TRILINOS
#    include <deal.II/lac/trilinos_parallel_block_vector.h>
#    include <deal.II/lac/trilinos_vector.h>
#  endif
#  ifdef DEAL_II_WITH_PETSC
#    include <deal.II/lac/petsc_block_vector.h>
#    include <deal.II/lac/petsc_vector.h>
#  endif

#  include <iomanip>
#  include <iostream>

DEAL_II_NAMESPACE_OPEN

  template <typename VectorType>
  Nonlinear_Solver<VectorType>::AdditionalData::AdditionalData(
    const SolverType       &solvertype,
    const SolutionStrategy &strategy,
    const unsigned int      maximum_non_linear_iterations,
    const double            function_tolerance,
    const double            step_tolerance,
    const bool              no_init_setup,
    const unsigned int      maximum_setup_calls,
    const double            maximum_newton_step,
    const double            dq_relative_error,
    const unsigned int      maximum_beta_failures,
    const unsigned int      anderson_subspace_size)
    : solvertype(solvertype)
    , strategy(strategy)
    , maximum_non_linear_iterations(maximum_non_linear_iterations)
    , function_tolerance(function_tolerance)
    , step_tolerance(step_tolerance)
    , no_init_setup(no_init_setup)
    , maximum_setup_calls(maximum_setup_calls)
    , maximum_newton_step(maximum_newton_step)
    , dq_relative_error(dq_relative_error)
    , maximum_beta_failures(maximum_beta_failures)
    , anderson_subspace_size(anderson_subspace_size)
  {}



  template <typename VectorType>
  void
  Nonlinear_Solver<VectorType>::AdditionalData::add_parameters(ParameterHandler &prm)
  {
    static std::string strategy_str("newton");
    prm.add_parameter("Solution strategy",
                      strategy_str,
                      "Choose among newton|linesearch|fixed_point|picard",
                      Patterns::Selection(
                        "newton|linesearch|fixed_point|picard"));
    prm.add_action("Solution strategy", [&](const std::string &value) {
      if (value == "newton")
        strategy = newton;
      else if (value == "linesearch")
        strategy = linesearch;
      else if (value == "fixed_point")
        strategy = fixed_point;
      else if (value == "picard")
        strategy = picard;
      else
        Assert(false, ExcInternalError());
    });
    prm.add_parameter("Maximum number of nonlinear iterations",
                      maximum_non_linear_iterations);
    prm.add_parameter("Function norm stopping tolerance", function_tolerance);
    prm.add_parameter("Scaled step stopping tolerance", step_tolerance);

    prm.enter_subsection("Newton parameters");
    prm.add_parameter("No initial matrix setup", no_init_setup);
    prm.add_parameter("Maximum iterations without matrix setup",
                      maximum_setup_calls);
    prm.add_parameter("Maximum allowable scaled length of the Newton step",
                      maximum_newton_step);
    prm.add_parameter("Relative error for different quotient computation",
                      dq_relative_error);
    prm.leave_subsection();

    prm.enter_subsection("Linesearch parameters");
    prm.add_parameter("Maximum number of beta-condition failures",
                      maximum_beta_failures);
    prm.leave_subsection();


    prm.enter_subsection("Fixed point and Picard parameters");
    prm.add_parameter("Anderson acceleration subspace size",
                      anderson_subspace_size);
    prm.leave_subsection();
  }


  namespace
  {
    template <typename VectorType>
    int
    residual_callback(N_Vector yy, N_Vector FF, void *user_data)
    {
      KINSOL<VectorType> &solver =
        *static_cast<KINSOL<VectorType> *>(user_data);
      GrowingVectorMemory<VectorType> mem;

      typename VectorMemory<VectorType>::Pointer src_yy(mem);
      solver.reinit_vector(*src_yy);

      typename VectorMemory<VectorType>::Pointer dst_FF(mem);
      solver.reinit_vector(*dst_FF);

      internal::copy(*src_yy, yy);

      int err = 0;
      if (solver.residual)
        err = solver.residual(*src_yy, *dst_FF);
      else
        Assert(false, ExcInternalError());

      internal::copy(FF, *dst_FF);

      return err;
    }



    template <typename VectorType>
    int
    iteration_callback(N_Vector yy, N_Vector FF, void *user_data)
    {
      KINSOL<VectorType> &solver =
        *static_cast<KINSOL<VectorType> *>(user_data);
      GrowingVectorMemory<VectorType> mem;

      typename VectorMemory<VectorType>::Pointer src_yy(mem);
      solver.reinit_vector(*src_yy);

      typename VectorMemory<VectorType>::Pointer dst_FF(mem);
      solver.reinit_vector(*dst_FF);

      internal::copy(*src_yy, yy);

      int err = 0;
      if (solver.iteration_function)
        err = solver.iteration_function(*src_yy, *dst_FF);
      else
        Assert(false, ExcInternalError());

      internal::copy(FF, *dst_FF);

      return err;
    }



    template <typename VectorType>
    int
    setup_jacobian_callback(N_Vector u,
                            N_Vector f,
                            SUNMatrix /* ignored */,
                            void *user_data,
                            N_Vector /* tmp1 */,
                            N_Vector /* tmp2 */)
    {
      // Receive the object that describes the linear solver and
      // unpack the pointer to the KINSOL object from which we can then
      // get the 'setup' function.
      const KINSOL<VectorType> &solver =
        *static_cast<const KINSOL<VectorType> *>(user_data);

      // Allocate temporary (deal.II-type) vectors into which to copy the
      // N_vectors
      GrowingVectorMemory<VectorType>            mem;
      typename VectorMemory<VectorType>::Pointer ycur(mem);
      typename VectorMemory<VectorType>::Pointer fcur(mem);
      solver.reinit_vector(*ycur);
      solver.reinit_vector(*fcur);

      internal::copy(*ycur, u);
      internal::copy(*fcur, f);

      // Call the user-provided setup function with these arguments:
      solver.setup_jacobian(*ycur, *fcur);

      return 0;
    }



    template <typename VectorType>
    int
    solve_with_jacobian_callback(SUNLinearSolver LS,
                                 SUNMatrix /*ignored*/,
                                 N_Vector x,
                                 N_Vector b,
                                 realtype tol)
    {
      // Receive the object that describes the linear solver and
      // unpack the pointer to the KINSOL object from which we can then
      // get the 'reinit' and 'solve' functions.
      const KINSOL<VectorType> &solver =
        *static_cast<const KINSOL<VectorType> *>(LS->content);

      // This is where we have to make a decision about which of the two
      // signals to call. Let's first check the more modern one:
      if (solver.solve_with_jacobian)
        {
          // Allocate temporary (deal.II-type) vectors into which to copy the
          // N_vectors
          GrowingVectorMemory<VectorType>            mem;
          typename VectorMemory<VectorType>::Pointer src_b(mem);
          typename VectorMemory<VectorType>::Pointer dst_x(mem);

          solver.reinit_vector(*src_b);
          solver.reinit_vector(*dst_x);

          internal::copy(*src_b, b);

          const int err = solver.solve_with_jacobian(*src_b, *dst_x, tol);

          internal::copy(x, *dst_x);

          return err;
        }
      else
        {
          // User has not provided the modern callback, so the fact that we are
          // here means that they must have given us something for the old
          // signal. Check this.
          Assert(solver.solve_jacobian_system, ExcInternalError());

          // Allocate temporary (deal.II-type) vectors into which to copy the
          // N_vectors
          GrowingVectorMemory<VectorType>            mem;
          typename VectorMemory<VectorType>::Pointer src_ycur(mem);
          typename VectorMemory<VectorType>::Pointer src_fcur(mem);
          typename VectorMemory<VectorType>::Pointer src_b(mem);
          typename VectorMemory<VectorType>::Pointer dst_x(mem);

          solver.reinit_vector(*src_b);
          solver.reinit_vector(*dst_x);

          internal::copy(*src_b, b);

          // Call the user-provided setup function with these arguments. Note
          // that Sundials 4.x and later no longer provide values for
          // src_ycur and src_fcur, and so we simply pass dummy vector in.
          // These vectors will have zero lengths because we don't reinit them
          // above.
          const int err =
            solver.solve_jacobian_system(*src_ycur, *src_fcur, *src_b, *dst_x);

          internal::copy(x, *dst_x);

          return err;
        }
    }
  } // namespace



  template <typename VectorType>
  KINSOL<VectorType>::KINSOL(const AdditionalData &data)
    : KINSOL(data, MPI_COMM_SELF)
  {}



  template <typename VectorType>
  Nonlinear_Solver<VectorType>::Nonlinear_Solver(
    const AdditionalData &data,
    const MPI_Comm &      mpi_comm)
    : data(data)
    , mpi_communicator(mpi_comm)
    , Nonlinear_Solver_mem(nullptr)
    , solution(nullptr)
    , u_scale(nullptr)
    , f_scale(nullptr)
  {
    set_functions_to_trigger_an_assert();
  }



  template <typename VectorType>
  KINSOL<VectorType>::~KINSOL()
  {
    std::cout << "idk what should be here" << std::endl;
  }



  template <typename VectorType>
  unsigned int
  Nonlinear_Solver<VectorType>::solve(VectorType &initial_guess_and_solution)
  {
    std::cout << "This function shouldn't be called because polymorphism" << std::endl;

    return 0;
  }

  template <typename VectorType>
  void
  Nonlinear_Solver<VectorType>::set_functions_to_trigger_an_assert()
  {
    reinit_vector = [](VectorType &) {
      AssertThrow(false, ExcFunctionNotProvided("reinit_vector"));
    };
  }

  template class Nonlinear_Solver<Vector<double>>;
  template class Nonlinear_Solver<BlockVector<double>>;

  template class Nonlinear_Solver<LinearAlgebra::distributed::Vector<double>>;
  template class Nonlinear_Solver<LinearAlgebra::distributed::BlockVector<double>>;

#  ifdef DEAL_II_WITH_MPI

#    ifdef DEAL_II_WITH_TRILINOS
  template class Nonlinear_Solver<TrilinosWrappers::MPI::Vector>;
  template class Nonlinear_Solver<TrilinosWrappers::MPI::BlockVector>;
#    endif

#    ifdef DEAL_II_WITH_PETSC
#      ifndef PETSC_USE_COMPLEX
  template class Nonlinear_Solver<PETScWrappers::MPI::Vector>;
  template class Nonlinear_Solver<PETScWrappers::MPI::BlockVector>;
#      endif
#    endif

#  endif

DEAL_II_NAMESPACE_CLOSE
