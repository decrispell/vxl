// This is bbas/bpgl/algo/bpgl_gridding.h
#ifndef bpgl_gridding_h_
#define bpgl_gridding_h_
//:
// \file
// \brief Transform irregular data to gridded 2D format (e.g. DSMs)
// \author Dan Crispell
// \date Nov 26, 2018
//

#include <iostream>
#include <stdexcept>
#include <vector>
#include <bvgl/bvgl_k_nearest_neighbors_2d.h>
#include <vnl/vnl_numeric_traits.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vil/vil_image_view.h>
#include <vnl/vnl_math.h>
#ifdef _MSC_VER
#  include <vcl_msvc_warnings.h>
#endif

namespace bpgl_gridding
{

template<class T, class DATA_T>
class inverse_distance_interp
{
public:
  inverse_distance_interp( T max_dist=vnl_numeric_traits<T>::maxval,
                           DATA_T invalid_val=DATA_T(NAN) )
    : max_dist_(max_dist), invalid_val_(invalid_val)
  {}
  DATA_T operator() ( vgl_point_2d<T> loc,
                      std::vector<vgl_point_2d<T>> const& neighbor_locs,
                      std::vector<DATA_T> const& neighbor_vals ) const
  {
    T weight_sum(0);
    T val_sum(0);
    const T eps(1e-6);
    const unsigned num_neighbors = neighbor_locs.size();
    for (unsigned i=0; i<num_neighbors; ++i) {
      T dist = (neighbor_locs[i] - loc).length();
      if (dist <= max_dist_) {
        if (dist < eps) {
          dist = eps;
        }
        T weight = 1.0 / dist;
        weight_sum += weight;
        val_sum += weight*neighbor_vals[i];
      }
    }
    if (weight_sum == T(0)) {
      return invalid_val_;
    }
    return val_sum / weight_sum;
  }
private:
  T max_dist_;
  DATA_T invalid_val_;
};


template<class T, class DATA_T>
class linear_interp
{
public:
  linear_interp( T max_dist=vnl_numeric_traits<T>::maxval,
                 DATA_T invalid_val=DATA_T(NAN),
                 T regularization_const=1e-1,
                 T dist_eps=1e-3)
    //: Linear interpolation function object constructor
    // max_dist: Neighboring points farther than max_dist will be ignored
    // invalid_val: Value to return when too few neighbors to interpolate
    // regularization_const: Larger regularization values will bias the solution towards "flatter" functions.  Very large values will result in weighted averages of neighbor values.
    // dist_eps: The smallest meaningful distance of input points. Must be > 0.
    : max_dist_(max_dist), invalid_val_(invalid_val),
    regularization_const_(regularization_const),
    dist_eps_(dist_eps)
  {}
  DATA_T operator() ( vgl_point_2d<T> loc,
                      std::vector<vgl_point_2d<T>> const& neighbor_locs,
                      std::vector<DATA_T> const& neighbor_vals ) const
  {
    T weight_sum(0);
    T val_sum(0);
    const unsigned num_neighbors = neighbor_locs.size();
    vnl_matrix<T> A(num_neighbors,3);
    vnl_vector<T> b(num_neighbors);
    int num_valid_neighbors = 0;
    // compute mean neighbor location
    T x_mean = 0, y_mean = 0, min_dist = max_dist_;
    DATA_T val_mean(0);
    std::vector<T> dists(num_neighbors);
    for (unsigned i=0; i<num_neighbors; ++i) {
      vgl_point_2d<T> const& neighbor_loc(neighbor_locs[i]);
      const T dist = (neighbor_loc - loc).length();
      dists[i] = dist;
      if (dist <= max_dist_) {
        if (dist < min_dist) {
          min_dist = dist;
        }
        ++num_valid_neighbors;
        x_mean += neighbor_loc.x();
        y_mean += neighbor_loc.y();
        val_mean += neighbor_vals[i];
      }
    }
    x_mean /= num_valid_neighbors;
    y_mean /= num_valid_neighbors;
    val_mean /= num_valid_neighbors;

    if (num_valid_neighbors < 1) {
      return invalid_val_;
    }
    for (unsigned i=0; i<num_neighbors; ++i) {
      const T weight = (min_dist + dist_eps_) / (dists[i] + dist_eps_);
      vgl_point_2d<T> const& neighbor_loc(neighbor_locs[i]);
      A[i][0] = weight * (neighbor_loc.x() - x_mean);
      A[i][1] = weight * (neighbor_loc.y() - y_mean);
      A[i][2] = weight;
      b[i] = weight * (neighbor_vals[i] - val_mean);
    }
    // employ Tikhonov Regularization to cope with degenerate point configurations
    vnl_matrix<T> R(3, 3, 0);
    // A function of the form z = ax + by + c is fit to the neighbor data points.
    // Apply regularization to the a and b parameters, but not c.  This way, the fitting
    // will approach a weighted average as the regularization constant is increased.
    R[0][0] = regularization_const_;
    R[1][1] = regularization_const_;
    R[2][2] = regularization_const_;
    vnl_matrix<T> A_transpose = A.transpose();
    vnl_vector<T> f = vnl_matrix_inverse<T>(A_transpose*A + R.transpose()*R)*A_transpose * b;
    std::cout << "f = " << f << std::endl;
    DATA_T value = f[0]*(loc.x() - x_mean) + f[1]*(loc.y() - y_mean) + f[2] + val_mean;
    return value;
  }
private:
  T max_dist_;
  DATA_T invalid_val_;
  T regularization_const_;
  T dist_eps_;
};



template<class T, class DATA_T, class INTERP_T>
vil_image_view<DATA_T>
grid_data_2d(std::vector<vgl_point_2d<T>> const& data_in_loc,
             std::vector<DATA_T> const& data_in,
             vgl_point_2d<T> out_upper_left,
             size_t out_ni, size_t out_nj,
             T step_size,
             INTERP_T &interp_fun,
             unsigned num_nearest_neighbors,
             double out_theta_radians=0.0)
{
  if (data_in_loc.size() != data_in.size()) {
    throw std::runtime_error("Input location and data arrays not equal size");
  }
  bvgl_k_nearest_neighbors_2d<T> knn(data_in_loc);

  vgl_vector_2d<T> i_vec(std::cos(out_theta_radians), std::sin(out_theta_radians));
  vgl_vector_2d<T> j_vec(std::sin(out_theta_radians), -std::cos(out_theta_radians));

  vil_image_view<DATA_T> gridded(out_ni, out_nj);
  for (unsigned j=0; j<out_nj; ++j) {
    for (unsigned i=0; i<out_ni; ++i) {
      vgl_point_2d<T> loc = out_upper_left +
        i*step_size*i_vec + j*step_size*j_vec;
      std::vector<vgl_point_2d<T> > neighbor_locs;
      vnl_vector<int> neighbor_inds(num_nearest_neighbors);
      if (!knn.knn(loc, num_nearest_neighbors, neighbor_locs, neighbor_inds)) {
        throw std::runtime_error("KNN failed to return neighbors");
        continue;
      }
      std::vector<DATA_T> neighbor_vals;
      for (auto nidx : neighbor_inds) {
        neighbor_vals.push_back(data_in[nidx]);
      }
      T val = interp_fun(loc, neighbor_locs, neighbor_vals);
      gridded(i,j) = val;
    }
  }
  return gridded;
}

 template<class pointT, class pixelT>
void  pointset_from_grid(vil_image_view<pixelT> const& grid, vgl_point_2d<pointT> const& upper_left, pointT step_size,
                         std::vector<vgl_point_3d<pointT> >& ptset, double out_theta_radians = 0.0){
  ptset.clear();
  vgl_vector_2d<pointT> i_vec(std::cos(out_theta_radians), std::sin(out_theta_radians));
  vgl_vector_2d<pointT> j_vec(std::sin(out_theta_radians), -std::cos(out_theta_radians));
  pointT xul = upper_left.x(), yul = upper_left.y();
  size_t ni = grid.ni(), nj = grid.nj();
  for(size_t j = 0; j<nj; ++j)
    for(size_t i = 0; i<ni; ++i){
      vgl_point_2d<pixelT> loc = upper_left +
        i*step_size*i_vec + j*step_size*j_vec;
      pointT z = grid(i,j);
      if(!vnl_math::isfinite(z))
        continue;
      ptset.emplace_back(loc.x(), loc.y(), z);
    }
}


}
#endif
