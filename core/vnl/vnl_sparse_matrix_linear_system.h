#ifndef vnl_sparse_matrix_linear_system_h_
#define vnl_sparse_matrix_linear_system_h_
#ifdef __GNUC__
#pragma interface
#endif
// This is vxl/vnl/vnl_sparse_matrix_linear_system.h

//: \file
//  \brief vnl_sparse_matrix -> vnl_linear_system adaptor
//  \author David Capel, capes@robots, July 2000
//  An adaptor that converts a vnl_sparse_matrix<T> to a vnl_linear_system
//

//  Modifications
//  LSB (Manchester) 19/3/01 Documentation tidied    
//
//-----------------------------------------------------------------------------

#include <vnl/vnl_linear_system.h>
#include <vnl/vnl_sparse_matrix.h>

//: vnl_sparse_matrix -> vnl_linear_system adaptor
//  An adaptor that converts a vnl_sparse_matrix<T> to a vnl_linear_system
template <class T>
class vnl_sparse_matrix_linear_system : public vnl_linear_system {
public:
  //::Constructor from vnl_sparse_matrix<double> for system Ax = b
  // Keeps a reference to the original sparse matrix A and vector b so DO NOT DELETE THEM!!
  vnl_sparse_matrix_linear_system(vnl_sparse_matrix<T> const& A, vnl_vector<T> const& b) :
    vnl_linear_system(A.columns(), A.rows()), A_(A), b_(b) {}

  //:  Implementations of the vnl_linear_system virtuals.
  void multiply(vnl_vector<double> const& x, vnl_vector<double> & b) const;
    //:  Implementations of the vnl_linear_system virtuals.
  void transpose_multiply(vnl_vector<double> const& b, vnl_vector<double> & x) const;
    //:  Implementations of the vnl_linear_system virtuals.
  void get_rhs(vnl_vector<double>& b) const;
  //:  Implementations of the vnl_linear_system virtuals.
  void apply_preconditioner(vnl_vector<double> const& x, vnl_vector<double> & px) const;

protected:
  vnl_sparse_matrix<T> const& A_;
  vnl_vector<T> const& b_;
  vnl_vector<double> jacobi_precond_;
};

#endif // vnl_sparse_matrix_linear_system_h_
