/* $Id$ */
/* Author: Wolfgang Bangerth, University of Heidelberg, 2001 */

				 // As usual, the program starts with
				 // a rather long list of include
				 // files which you are probably
				 // already used to by now:
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <base/logstream.h>
#include <lac/vector.h>
#include <lac/sparse_matrix.h>
#include <lac/solver_cg.h>
#include <lac/vector_memory.h>
#include <lac/precondition.h>
#include <grid/tria.h>
#include <grid/grid_generator.h>
#include <grid/tria_boundary_lib.h>
#include <grid/tria_accessor.h>
#include <grid/tria_iterator.h>
#include <dofs/dof_handler.h>
#include <dofs/dof_constraints.h>
#include <dofs/dof_accessor.h>
#include <dofs/dof_tools.h>
#include <fe/fe_q.h>
#include <fe/fe_values.h>
#include <fe/mapping_q.h>
#include <numerics/vectors.h>
#include <numerics/matrices.h>

				 // Just this one is new: it declares
				 // a class
				 // ``CompressedSparsityPattern'',
				 // which we will use and explain
				 // further down below.
#include <lac/compressed_sparsity_pattern.h>

				 // We will make use of the std::find
				 // algorithm of the C++ standard
				 // library, so we have to include the
				 // following file for its
				 // declaration:
#include <algorithm>



				 // Then we declare a class which
				 // represents the solution of a
				 // Laplace problem. As this example
				 // program is based on step-5, the
				 // class looks rather the same, with
				 // the sole structural difference
				 // that the functions
				 // ``assemble_system'' now calls
				 // ``solve'' itself, and is thus
				 // called ``assemble_and_solve'', and
				 // that the output function was
				 // dropped since the solution
				 // function is so boring that it is
				 // not worth being viewed.
				 //
				 // The only other noteworthy change
				 // is that the constructor takes a
				 // value representing the polynomial
				 // degree of the mapping to be used
				 // later on, and that it has another
				 // member variable representing
				 // exactly this mapping. In general,
				 // this variable will occur in real
				 // applications at the same places
				 // where the finite element is
				 // declared or used.
template <int dim>
class LaplaceProblem 
{
  public:
    LaplaceProblem (const unsigned int mapping_degree);
    void run ();
    
  private:
    void setup_system ();
    void assemble_and_solve ();
    void solve ();

    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;
    MappingQ<dim>        mapping;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    ConstraintMatrix     mean_value_constraints;

    Vector<double>       solution;
    Vector<double>       system_rhs;
};



				 // Construct such an object, by
				 // initializing the variables. Here,
				 // we use linear finite elements (the
				 // argument to the ``fe'' variable
				 // denotes the polynomial degree),
				 // and mappings of given order. Print
				 // to screen what we are about to do.
template <int dim>
LaplaceProblem<dim>::LaplaceProblem (const unsigned int mapping_degree) :
                fe (1),
		dof_handler (triangulation),
		mapping (mapping_degree)
{
  std::cout << "Using mapping with degree " << mapping_degree << ":"
	    << std::endl
	    << "============================"
	    << std::endl;
};



				 // The first task is to set up the
				 // variables for this problem. This
				 // includes generating a valid
				 // ``DoFHandler'' object, as well as
				 // the sparsity patterns for the
				 // matrix, and the object
				 // representing the constraints that
				 // the mean value of the degrees of
				 // freedom on the boundary be zero.
template <int dim>
void LaplaceProblem<dim>::setup_system ()
{
				   // The first task is trivial:
				   // generate an enumeration of the
				   // degrees of freedom:
  dof_handler.distribute_dofs (fe);

				   // Next task is to construct the
				   // object representing the
				   // constraint that the mean value
				   // of the degrees of freedom on the
				   // boundary shall be zero. For
				   // this, we first want a list of
				   // those nodes which are actually
				   // at the boundary. The
				   // ``DoFTools'' class has a
				   // function that returns an array
				   // of boolean values where ``true''
				   // indicates that the node is at
				   // the boundary. The second
				   // argument denotes a mask
				   // selecting which components of
				   // vector valued finite elements we
				   // want to be considered. Since we
				   // have a scalar finite element
				   // anyway, this mask consists of
				   // only one entry, and its value
				   // must be ``true''.
  std::vector<bool> boundary_dofs (dof_handler.n_dofs(), false);
  DoFTools::extract_boundary_dofs (dof_handler, std::vector<bool>(1,true),
				   boundary_dofs);
  
				   // Let us first pick out the first
				   // boundary node from this list. We
				   // do that by searching for the
				   // first ``true'' value in the
				   // array (note that ``std::find''
				   // returns an iterator to this
				   // element), and computing its
				   // distance to the overall first
				   // element in the array to get its
				   // index:
  const unsigned int first_boundary_dof
    = std::distance (std::find (boundary_dofs.begin(),
				boundary_dofs.end(),
				true),
		     boundary_dofs.begin());
	
  mean_value_constraints.clear ();
  mean_value_constraints.add_line (first_boundary_dof);
  for (unsigned int i=first_boundary_dof+1; i<dof_handler.n_dofs(); ++i)
    if (boundary_dofs[i] == true)
      mean_value_constraints.add_entry (first_boundary_dof,
					i, -1);
  mean_value_constraints.close ();

  CompressedSparsityPattern csp (dof_handler.n_dofs(),
				 dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, csp);
  mean_value_constraints.condense (csp);

  sparsity_pattern.copy_from (csp);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
};



				 // The next function then assembles
				 // the linear system of equations,
				 // solves it, and evaluates the
				 // solution. This then makes three
				 // actions, and we will put them into
				 // eight true statements (excluding
				 // declaration of variables, and
				 // handling of temporary
				 // vectors). Thus, this function is
				 // something for the very
				 // lazy. Nevertheless, the functions
				 // called are rather powerful, and
				 // through them this function uses a
				 // good deal of the whole
				 // library. But let's look at each of
				 // the steps.
template <int dim>
void LaplaceProblem<dim>::assemble_and_solve () 
{

				   // First, we have to assemble the
				   // matrix and the right hand
				   // side. In all previous examples,
				   // we have investigated various
				   // ways how to do this
				   // manually. However, since the
				   // Laplace matrix and simple right
				   // hand sides appear so frequently
				   // in applications, the library
				   // provides functions for actually
				   // doing this for you, i.e. they
				   // perform the loop over all cells,
				   // setting up the local matrices
				   // and vectors, and putting them
				   // together for the end result.
				   //
				   // The following are the two most
				   // commonly used ones: creation of
				   // the Laplace matrix and creation
				   // of a right hand side vector from
				   // body or boundary forces. They
				   // take the mapping object, the
				   // ``DoFHandler'' object
				   // representing the degrees of
				   // freedom and the finite element
				   // in use, a quadrature formula to
				   // be used, and the output
				   // object. The function that
				   // creates a right hand side vector
				   // also has to take a function
				   // object describing the
				   // (continuous) right hand side
				   // function.
				   //
				   // Let us look at the way the
				   // matrix and body forces are
				   // integrated:
  const unsigned int gauss_degree
    = std::max (static_cast<unsigned int>(ceil(1.*(mapping.get_degree()+1)/2)),
		2U);
  MatrixTools::create_laplace_matrix (mapping, dof_handler,
				      QGauss<dim>(gauss_degree),
				      system_matrix);
  VectorTools::create_right_hand_side (mapping, dof_handler,
				       QGauss<dim>(gauss_degree),
				       ConstantFunction<dim>(-2),
				       system_rhs);
				   // That's quite simple, right?
				   //
				   // Two remarks are in order,
				   // though: First, these functions
				   // are used in a lot of
				   // contexts. Maybe you want to
				   // create a Laplace or mass matrix
				   // for a vector values finite
				   // element; or you want to use the
				   // default Q1 mapping; or you want
				   // to assembled the matrix with a
				   // coefficient in the Laplace
				   // operator. For this reason, there
				   // are quite a large number of
				   // variants of these functions in
				   // the ``MatrixCreator'' and
				   // ``MatrixTools''
				   // classes. Whenever you need a
				   // slighly different version of
				   // these functions than the ones
				   // called above, it is certainly
				   // worthwhile to take a look at the
				   // documentation and to check
				   // whether something fits your
				   // needs.
				   //
				   // The second remark concerns the
				   // quadrature formula we use: we
				   // want to integrate over bilinear
				   // shape functions, so we know that
				   // we have to use at least a Gauss2
				   // quadrature formula. On the other
				   // hand, we want to have the
				   // quadrature rule to have at least
				   // the order of the boundary
				   // approximation. Since the order
				   // of Gauss-r is 2r, and the order
				   // of the boundary approximation
				   // using polynomials of degree p is
				   // p+1, we know that 2r>=p+1. Since
				   // r has to be an integer and (as
				   // mentioned above) has to be at
				   // least 2, this makes up for the
				   // formula above computing
				   // ``gauss_degree''.
				   //
				   // Note also, that we have used a
				   // class called ``QGauss''. By now,
				   // we have only used ``QGauss4'',
				   // or the like, which implement a
				   // Gauss quadrature rule of fixed
				   // order. The ``QGauss'' class is
				   // more general, taking a parameter
				   // which indicates of which degree
				   // it shall be; for small degrees,
				   // the object then parallels
				   // objects of type ``QGaussR'' with
				   // fixed R, but it also provides
				   // quadrature rules of higher
				   // degree which are no longer
				   // hardcoded in the library.

				   // Since the generation of the body
				   // force contributions to the right
				   // hand side vector was so simple,
				   // we do that all over again for
				   // the boundary forces as well:
				   // allocate a vector of the right
				   // size and call the right
				   // function. The boundary function
				   // has constant values, so we can
				   // generate an object from the
				   // library on the fly, and we use
				   // the same quadrature formula as
				   // above, but this time of lower
				   // dimension since we integrate
				   // over faces now instead of cells:
  Vector<double> tmp (system_rhs.size());
  VectorTools::create_boundary_right_hand_side (mapping, dof_handler,
						QGauss<dim-1>(gauss_degree),
						ConstantFunction<dim>(1),
						tmp);
				   // Then add the contributions from
				   // the boundary to those from the
				   // interior of the domain:
  system_rhs += tmp;
				   // For assembling the right hand
				   // side, we had to use two
				   // different vector objects, and
				   // later add them together. The
				   // reason we had to do so is that
				   // the
				   // ``VectorTools::create_right_hand_side''
				   // and
				   // ``VectorTools::create_boundary_right_hand_side''
				   // functions first clear the output
				   // vector, rather than adding up
				   // their results to previous
				   // contents. This can reasonably be
				   // called a design flaw in the
				   // library made in its infancy, but
				   // unfortunately things are as they
				   // are for some time now and it is
				   // difficult to change such things
				   // that silently break existing
				   // code, so we have to live with
				   // that.

				   // Now, the linear system is set
				   // up, so we can eliminate the one
				   // degree of freedom which we
				   // constrained to the other DoFs on
				   // the boundary for the mean value
				   // constraint from matrix and right
				   // hand side vector, and solve the
				   // system. After that, distribute
				   // the constraints again, which in
				   // this case means setting the
				   // constrained degree of freedom to
				   // its proper value
  mean_value_constraints.condense (system_matrix);
  mean_value_constraints.condense (system_rhs);  

  solve ();
  mean_value_constraints.distribute (solution);

				   // Finally, evaluate what we got as
				   // solution. As stated in the
				   // introduction, we are interested
				   // in the H1 seminorm of the
				   // solution. Here, as well, we have
				   // a function in the library that
				   // does this, although in a
				   // slightly non-obvious way: the
				   // ``VectorTools::integrate_difference''
				   // function integrates the norm of
				   // the difference between a finite
				   // element function and a
				   // continuous function. If we
				   // therefore want the norm of a
				   // finite element field, we just
				   // put the continuous function to
				   // zero. Note that this function,
				   // just as so many other ones in
				   // the library as well, has at
				   // least two versions, one which
				   // takes a mapping as argument
				   // (which we make us of here), and
				   // the one which we have used in
				   // previous examples which
				   // implicitely uses ``MappingQ1''.
				   // Also note that we take a
				   // quadrature formula of one degree
				   // higher, in order to avoid
				   // superconvergence effects where
				   // the solution happens to be
				   // especially close to the exact
				   // solution at certain points (we
				   // don't know whether this might be
				   // the case here, but there are
				   // cases known of this, and we just
				   // want to make sure):
  Vector<float> norm_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (mapping, dof_handler,
				     solution,
				     ZeroFunction<dim>(),
				     norm_per_cell,
				     QGauss<dim>(gauss_degree+1),
				     H1_seminorm);
				   // Then, the function just called
				   // returns its results as a vector
				   // of values each of which denotes
				   // the norm on one cell. To get the
				   // global norm, a simple
				   // computation shows that we have
				   // to take the l2 norm of the
				   // vector:
  const double norm = norm_per_cell.l2_norm();

				   // Last task -- show output:
  std::cout << "  " << triangulation.n_active_cells() << " cells:  "
	    << "  |u|_1="
	    << norm
	    << ", error="
	    << fabs(norm-sqrt(3.14159265358/2))
	    << std::endl;
};



				 // The following function solving the
				 // linear system of equations is
				 // copied from step-5 and is
				 // explained there in some detail:
template <int dim>
void LaplaceProblem<dim>::solve () 
{
  SolverControl           solver_control (1000, 1e-12);
  PrimitiveVectorMemory<> vector_memory;
  SolverCG<>              cg (solver_control, vector_memory);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  cg.solve (system_matrix, solution, system_rhs,
	    preconditioner);
};



				 // Finally the main function
				 // controlling the different steps to
				 // be performed. Its content is
				 // rather straightforward, generating
				 // a triangulation of a circle,
				 // associating a boundary to it, and
				 // then doing several cycles on
				 // subsequently finer grids. Note
				 // again that we have put mesh
				 // refinement into the loop header;
				 // this may be something for a test
				 // program, but for real applications
				 // you should consider that this
				 // implies that the mesh is refined
				 // after the loop is executed the
				 // last time since the increment
				 // clause (the last part of the
				 // three-parted loop header) is
				 // executed before the comparison
				 // part (the second one), which may
				 // be rather costly if the mesh is
				 // already quite refined. In that
				 // case, you should arrange code such
				 // that the mesh is not further
				 // refined after the last loop run
				 // (or you should do it at the
				 // beginning of each run except for
				 // the first one).
template <int dim>
void LaplaceProblem<dim>::run () 
{
  GridGenerator::hyper_ball (triangulation);
  static const HyperBallBoundary<dim> boundary;
  triangulation.set_boundary (0, boundary);
  
  for (unsigned int cycle=0; cycle<6; ++cycle, triangulation.refine_global(1))
    {
      setup_system ();
      assemble_and_solve ();
    };
};

    

				 // Finally the main function. It's
				 // structure is the same as that used
				 // in several of the previous
				 // examples, so probably needs no
				 // more explanation.
int main () 
{
  try
    {
      deallog.depth_console (0);
      std::cout.precision(5);

				       // This is the main loop, doing
				       // the computations with
				       // mappings of linear through
				       // cubic mappings. Note that
				       // since we need the object of
				       // type ``LaplaceProblem<2>''
				       // only once, we do not even
				       // name it, but create an
				       // unnamed such object and call
				       // the ``run'' function of it,
				       // subsequent to which it is
				       // immediately destroyed again.
      for (unsigned int mapping_degree=1; mapping_degree<=3; ++mapping_degree)
	LaplaceProblem<2>(mapping_degree).run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Exception on processing: " << std::endl
		<< exc.what() << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      return 1;
    }
  catch (...) 
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Unknown exception!" << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      return 1;
    };

  return 0;
};
