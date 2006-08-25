//----------------------------  project_common.cc  ---------------------------
//    $Id: project_common.cc 12732 2006-03-28 23:15:45Z wolf $
//    Version: $Name$ 
//
//    Copyright (C) 2006 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------  project_common.cc  ---------------------------


// common framework to check whether an element of polynomial order p can
// represent functions of order q

#include "../tests.h"
#include <base/function.h>
#include <base/logstream.h>
#include <base/quadrature_lib.h>
#include <lac/vector.h>

#include <grid/tria.h>
#include <dofs/dof_handler.h>
#include <dofs/dof_constraints.h>
#include <grid/grid_generator.h>
#include <grid/grid_refinement.h>
#include <grid/tria_accessor.h>
#include <grid/tria_iterator.h>
#include <grid/tria_boundary_lib.h>
#include <dofs/dof_accessor.h>
#include <dofs/dof_tools.h>
#include <numerics/vectors.h>
#include <fe/fe_abf.h>
#include <fe/fe_dgp.h>
#include <fe/fe_dgp_monomial.h>
#include <fe/fe_dgp_nonparametric.h>
#include <fe/fe_dgq.h>
#include <fe/fe_nedelec.h>
#include <fe/fe_q.h>
#include <fe/fe_q_hierarchical.h>
#include <fe/fe_raviart_thomas.h>
#include <fe/fe_system.h>

#include <fstream>
#include <vector>


template <int dim>
void test ();


template <int dim>
class F :  public Function<dim>
{
  public:
    F (const unsigned int q,
       const unsigned int n_components)
		    :
		    Function<dim>(n_components),
		    q(q)
      {}
    
    virtual double value (const Point<dim> &p,
			  const unsigned int component) const
      {
	Assert ((component == 0) && (this->n_components == 1),
		ExcInternalError());
	double val = 0;
	for (unsigned int d=0; d<dim; ++d)
	  for (unsigned int i=0; i<=q; ++i)
	    val += (d+1)*(i+1)*std::pow (p[d], 1.*i);
	return val;
      }


    virtual void vector_value (const Point<dim> &p,
			       Vector<double>   &v) const
      {
	for (unsigned int c=0; c<v.size(); ++c)
	  {
	    v(c) = 0;
	    for (unsigned int d=0; d<dim; ++d)
	      for (unsigned int i=0; i<=q; ++i)
		v(c) += (d+1)*(i+1)*std::pow (p[d], 1.*i)+c;
	  }
      }
    
  private:
    const unsigned int q;
};



DeclException1 (ExcFailedProjection,
		double,
		<< "The projection was supposed to exactly represent the "
		<< "original function, but the relative residual is "
		<< arg1);


template <int dim>
void do_project (const Triangulation<dim> &triangulation,
		 const FiniteElement<dim> &fe,
		 const unsigned int        p,
		 const unsigned int        order_difference)
{  
  DoFHandler<dim>        dof_handler(triangulation);
  dof_handler.distribute_dofs (fe);

  deallog << "n_dofs=" << dof_handler.n_dofs() << std::endl;

  ConstraintMatrix constraints;
  DoFTools::make_hanging_node_constraints (dof_handler,
					   constraints);
  constraints.close ();

  Vector<double> projection (dof_handler.n_dofs());
  Vector<float>  error (triangulation.n_active_cells());
  for (unsigned int q=0; q<=p+2-order_difference; ++q)
    {
				       // project the function
      VectorTools::project (dof_handler,
			    constraints,
			    QGauss<dim>(p+2),
			    F<dim> (q, fe.n_components()),
			    projection);
				       // just to make sure it doesn't get
				       // forgotten: handle hanging node
				       // constraints
      constraints.distribute (projection);
      
				       // then compute the interpolation error
      VectorTools::integrate_difference (dof_handler,
					 projection,
					 F<dim> (q, fe.n_components()),
					 error,
					 QGauss<dim>(std::max(p,q)+1),
					 VectorTools::L2_norm);
      deallog << fe.get_name() << ", P_" << q
	      << ", rel. error=" << error.l2_norm() / projection.l2_norm()
	      << std::endl;
	  
      if (q<=p-order_difference)
	Assert (error.l2_norm() <= 1e-10*projection.l2_norm(),
		ExcFailedProjection(error.l2_norm() / projection.l2_norm()));
    }
}



// check the given element of polynomial order p. the last parameter, if
// given, denotes a gap in convergence order; for example, the Nedelec element
// of polynomial degree p has normal components of degree p-1 and therefore
// can only represent polynomials of degree p-1 exactly. the gap is then 1.
template <int dim>
void test_no_hanging_nodes (const FiniteElement<dim> &fe,
			    const unsigned int        p,
			    const unsigned int        order_difference = 0)
{
  Triangulation<dim>     triangulation;
  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global (3);

  do_project (triangulation, fe, p, order_difference);
}



// same test as above, but this time with a mesh that has hanging nodes
template <int dim>
void test_with_hanging_nodes (const FiniteElement<dim> &fe,
			      const unsigned int        p,
			      const unsigned int        order_difference = 0)
{
  Triangulation<dim>     triangulation;
  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global (1);
  triangulation.begin_active()->set_refine_flag ();
  triangulation.execute_coarsening_and_refinement ();
  triangulation.refine_global (1);
  
  do_project (triangulation, fe, p, order_difference);
}



// test with a 3d grid that has cells with face_orientation==false and hanging
// nodes. this trips up all sorts of pieces of code, for example there was a
// crash when computing hanging node constraints on such faces (see
// bits/face_orientation_crash), and it triggers all sorts of other
// assumptions that may be hidden in places
//
// the mesh we use is the 7 cells of the hyperball mesh in 3d, with each of
// the cells refined in turn. that then makes 7 meshes with 14 active cells
// each. this also cycles through all possibilities of coarser or finer cell
// having face_orientation==false
template <int dim>
void test_with_wrong_face_orientation (const FiniteElement<dim> &fe,
				       const unsigned int        p,
				       const unsigned int        order_difference = 0)
{
  if (dim != 3)
    return;
  
  for (unsigned int i=0; i<7; ++i)
    {
      Triangulation<dim>     triangulation;
      GridGenerator::hyper_ball (triangulation);
      typename Triangulation<dim>::active_cell_iterator
	cell = triangulation.begin_active();
      std::advance (cell, i);
      cell->set_refine_flag ();
      triangulation.execute_coarsening_and_refinement ();
  
      do_project (triangulation, fe, p, order_difference);
    }
}




// test with a 2d mesh that forms a square but subdivides it into 3
// elements. this tests the case of the sign_change thingy in
// fe_poly_tensor.cc
template <int dim>
void test_with_2d_deformed_mesh (const FiniteElement<dim> &fe,
				 const unsigned int        p,
				 const unsigned int        order_difference = 0)
{
  if (dim != 2)
    return;
  
  std::vector<Point<dim> > points_glob;
  std::vector<Point<dim> > points;

  points_glob.push_back (Point<dim> (0.0, 0.0));
  points_glob.push_back (Point<dim> (1.0, 0.0));
  points_glob.push_back (Point<dim> (1.0, 0.5));
  points_glob.push_back (Point<dim> (1.0, 1.0));
  points_glob.push_back (Point<dim> (0.6, 0.5));
  points_glob.push_back (Point<dim> (0.5, 1.0));
  points_glob.push_back (Point<dim> (0.0, 1.0));

  				   // Prepare cell data
  std::vector<CellData<dim> > cells (3);

  cells[0].vertices[0] = 0;
  cells[0].vertices[1] = 1;
  cells[0].vertices[2] = 4;
  cells[0].vertices[3] = 2;
  cells[0].material_id = 0;

  cells[1].vertices[0] = 4;
  cells[1].vertices[1] = 2;
  cells[1].vertices[2] = 5;
  cells[1].vertices[3] = 3;
  cells[1].material_id = 0;

  cells[2].vertices[0] = 0;
  cells[2].vertices[1] = 4;
  cells[2].vertices[2] = 6;
  cells[2].vertices[3] = 5;
  cells[2].material_id = 0;

  Triangulation<dim>     triangulation;
  triangulation.create_triangulation (points_glob, cells, SubCellData());
  
  do_project (triangulation, fe, p, order_difference);
}



// same as test_with_2d_deformed_mesh, but refine each element in turn. this
// makes sure we also check the sign_change thingy for refined cells
template <int dim>
void test_with_2d_deformed_refined_mesh (const FiniteElement<dim> &fe,
					 const unsigned int        p,
					 const unsigned int        order_difference = 0)
{
  if (dim != 2)
    return;

  for (unsigned int i=0; i<3; ++i)
    {
      std::vector<Point<dim> > points_glob;
      std::vector<Point<dim> > points;

      points_glob.push_back (Point<dim> (0.0, 0.0));
      points_glob.push_back (Point<dim> (1.0, 0.0));
      points_glob.push_back (Point<dim> (1.0, 0.5));
      points_glob.push_back (Point<dim> (1.0, 1.0));
      points_glob.push_back (Point<dim> (0.6, 0.5));
      points_glob.push_back (Point<dim> (0.5, 1.0));
      points_glob.push_back (Point<dim> (0.0, 1.0));

				       // Prepare cell data
      std::vector<CellData<dim> > cells (3);

      cells[0].vertices[0] = 0;
      cells[0].vertices[1] = 1;
      cells[0].vertices[2] = 4;
      cells[0].vertices[3] = 2;
      cells[0].material_id = 0;

      cells[1].vertices[0] = 4;
      cells[1].vertices[1] = 2;
      cells[1].vertices[2] = 5;
      cells[1].vertices[3] = 3;
      cells[1].material_id = 0;

      cells[2].vertices[0] = 0;
      cells[2].vertices[1] = 4;
      cells[2].vertices[2] = 6;
      cells[2].vertices[3] = 5;
      cells[2].material_id = 0;

      Triangulation<dim>     triangulation;
      triangulation.create_triangulation (points_glob, cells, SubCellData());

      switch (i)
	{
	  case 0:
		triangulation.begin_active()->set_refine_flag();
		break;
	  case 1:
		(++(triangulation.begin_active()))->set_refine_flag();
		break;
	  case 2:
		(++(++(triangulation.begin_active())))->set_refine_flag();
		break;
	  default:
		Assert (false, ExcNotImplemented());
	}
      
      do_project (triangulation, fe, p, order_difference);
    }
}



int main ()
{
  std::ofstream logfile(logname);
  logfile.precision (3);
  
  deallog.attach(logfile);
  deallog.depth_console(0);
  deallog.threshold_double(1.e-10);

  test<1>();
  test<2>();
  test<3>();
}

