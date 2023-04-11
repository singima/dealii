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

#include <deal.II/numerics/nonlinear.h>

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
  Nonlinear_Solver<VectorType>::AdditionalData_NL::AdditionalData_NL(
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
  Nonlinear_Solver<VectorType>::_NL::add_parameters(ParameterHandler &prm)
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
    residual_callback()
    {
      std::cout << "polymorphism...?" << std::endl;

      return 0;
    }



    template <typename VectorType>
    int
    iteration_callback()
    {
      std::cout << "polymorphism...?" << std::endl;

      return 0;
    }



    template <typename VectorType>
    int
    setup_jacobian_callback()
    {
      std::cout << "polymorphism...?" << std::endl;

      return 0;
    }



    template <typename VectorType>
    int
    solve_with_jacobian_callback()
    {
      std::cout << "polymorphism...?" << std::endl;

      return 0;
    }
  } // namespace



  template <typename VectorType>
  Nonlinear_Solver<VectorType>::Nonlinear_Solver(const AdditionalData_NL &data)
    : Nonlinear_Solver(data, MPI_COMM_SELF)
  {}



  // template <typename VectorType>
  // Nonlinear_Solver<VectorType>::Nonlinear_Solver(
  //   const AdditionalData &data,
  //   const MPI_Comm &      mpi_comm)
  //   : data(data)
  //   , mpi_communicator(mpi_comm)
  //   , Nonlinear_Solver_mem(nullptr)
  //   , solution(nullptr)
  //   , u_scale(nullptr)
  //   , f_scale(nullptr)
  // {
  //   set_functions_to_trigger_an_assert();
  // }



  // template <typename VectorType>
  // Nonlinear_Solver<VectorType>::~Nonlinear_Solver()
  // {
  //   std::cout << "idk what should be here" << std::endl;
  // }



//   template <typename VectorType>
//   unsigned int
//   Nonlinear_Solver<VectorType>::solve(VectorType &initial_guess_and_solution)
//   {
//     std::cout << "This function shouldn't be called because polymorphism" << std::endl;

//     return 0;
//   }

//   template <typename VectorType>
//   void
//   Nonlinear_Solver<VectorType>::set_functions_to_trigger_an_assert()
//   {
//     reinit_vector = [](VectorType &) {
//       AssertThrow(false, ExcFunctionNotProvided("reinit_vector"));
//     };
//   }

//   template class Nonlinear_Solver<Vector<double>>;
//   template class Nonlinear_Solver<BlockVector<double>>;

//   template class Nonlinear_Solver<LinearAlgebra::distributed::Vector<double>>;
//   template class Nonlinear_Solver<LinearAlgebra::distributed::BlockVector<double>>;

// #  ifdef DEAL_II_WITH_MPI

// #    ifdef DEAL_II_WITH_TRILINOS
//   template class Nonlinear_Solver<TrilinosWrappers::MPI::Vector>;
//   template class Nonlinear_Solver<TrilinosWrappers::MPI::BlockVector>;
// #    endif

// #    ifdef DEAL_II_WITH_PETSC
// #      ifndef PETSC_USE_COMPLEX
//   template class Nonlinear_Solver<PETScWrappers::MPI::Vector>;
//   template class Nonlinear_Solver<PETScWrappers::MPI::BlockVector>;
// #      endif
// #    endif

// #  endif

DEAL_II_NAMESPACE_CLOSE
