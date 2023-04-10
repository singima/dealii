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


#  include <deal.II/base/conditional_ostream.h>
#  include <deal.II/base/exceptions.h>
#  include <deal.II/base/logstream.h>
#  include <deal.II/base/mpi_stub.h>
#  include <deal.II/base/parameter_handler.h>

#  include <deal.II/lac/vector.h>
#  include <deal.II/lac/vector_memory.h>

#  include <boost/signals2.hpp>

#  include <nvector/nvector_serial.h>

#  include <memory>


DEAL_II_NAMESPACE_OPEN


template <typename VectorType = Vector<double>>
class Nonlinear_Solver
{
public:

/**
 * Additional parameters that can be passed to the Nonlinear_Solver class.
 */
class AdditionalData
{
public:
    enum SolutionStrategy
    {
    /**
     * Standard Newton iteration.
     */
    newton = KIN_NONE,
    /**
     * Newton iteration with linesearch.
     */
    linesearch = KIN_LINESEARCH,
    /**
     * Fixed point iteration.
     */
    fixed_point = KIN_FP,
    /**
     * Picard iteration.
     */
    picard = KIN_PICARD,
    };

    enum SolverType
    {
    /**
     * KINSOL, part of the SUNDIALS package
     */
    kinsol,
    /**
     * NOX, part of the Trilinos package
     */
    nox
    };

    
    AdditionalData(const SolverType &solvertype = kinsol,
                    const SolutionStrategy &strategy = linesearch,
                    const unsigned int maximum_non_linear_iterations = 200,
                    const double       function_tolerance            = 0.0,
                    const double       step_tolerance                = 0.0,
                    const bool         no_init_setup                 = false,
                    const unsigned int maximum_setup_calls           = 0,
                    const double       maximum_newton_step           = 0.0,
                    const double       dq_relative_error             = 0.0,
                    const unsigned int maximum_beta_failures         = 0,
                    const unsigned int anderson_subspace_size        = 0);

    
    void
    add_parameters(ParameterHandler &prm);

    SolverType solvertype;

    SolutionStrategy strategy;

    unsigned int maximum_non_linear_iterations;

    double function_tolerance;

    double step_tolerance;

    bool no_init_setup;

    unsigned int maximum_setup_calls;

    double maximum_newton_step;

    double dq_relative_error;

    unsigned int maximum_beta_failures;

    unsigned int anderson_subspace_size;
};

// Nonlinear_Solver(const AdditionalData &data = AdditionalData());

// Nonlinear_Solver(const AdditionalData &data, const MPI_Comm &mpi_comm);

// ~Nonlinear_Solver();

// unsigned int
// solve(VectorType &initial_guess_and_solution);

// std::function<void(VectorType &)> reinit_vector;

// std::function<int(const VectorType &src, VectorType &dst)> residual;

// std::function<int(const VectorType &src, VectorType &dst)>
//     iteration_function;

// std::function<int(const VectorType &current_u, const VectorType &current_f)>
//     setup_jacobian;

// DEAL_II_DEPRECATED
// std::function<int(const VectorType &ycur,
//                     const VectorType &fcur,
//                     const VectorType &rhs,
//                     VectorType &      dst)>
//     solve_jacobian_system;

// std::function<
//     int(const VectorType &rhs, VectorType &dst, const double tolerance)>
//     solve_with_jacobian;

// std::function<VectorType &()> get_solution_scaling;

// std::function<VectorType &()> get_function_scaling;

// // DeclException1(ExcKINSOLError,
// //                 int,
// //                 << "One of the SUNDIALS KINSOL internal functions "
// //                 << "returned a negative error code: " << arg1
// //                 << ". Please consult SUNDIALS manual.");


private:

DeclException1(ExcFunctionNotProvided,
                std::string,
                << "Please provide an implementation for the function \""
                << arg1 << "\"");

void
set_functions_to_trigger_an_assert();

AdditionalData data;

// MPI_Comm mpi_communicator;

// void *Nonlinear_Solver_mem;

// N_Vector solution;

// N_Vector u_scale;

// N_Vector f_scale;

// GrowingVectorMemory<VectorType> mem;
};


DEAL_II_NAMESPACE_CLOSE