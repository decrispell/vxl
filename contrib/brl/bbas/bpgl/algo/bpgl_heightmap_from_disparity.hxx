// This is contrib/brl/bbas/bpgl/algo/bpgl_heightmap_from_disparity.hxx
#ifndef bpgl_heightmap_from_disparity_hxx_
#define bpgl_heightmap_from_disparity_hxx_

#include "bpgl_3d_from_disparity.h"

#include "bpgl_heightmap_from_disparity.h"

#include <vnl/vnl_math.h>
#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_vector_3d.h>
#include <vbl/vbl_array_2d.h>

#include <vgl/vgl_triangle_scan_iterator.h>

//: helper function to compute barycentric coordinates
void compute_barycentric(vgl_point_2d<double> const& a,
                         vgl_point_2d<double> const& b,
                         vgl_point_2d<double> const& c,
                         vgl_point_2d<double> const& p,
                         double &u, double &v, double &w)
{
  vgl_vector_2d<double> v0 = b - a;
  vgl_vector_2d<double> v1 = c - a;
  vgl_vector_2d<double> v2 = p - a;
  double d00 = dot_product(v0, v0);
  double d01 = dot_product(v0, v1);
  double d11 = dot_product(v1, v1);
  double d20 = dot_product(v2, v0);
  double d21 = dot_product(v2, v1);
  double denom = d00 * d11 - d01 * d01;
  v = (d11 * d20 - d01 * d21) / denom;
  w = (d00 * d21 - d01 * d20) / denom;
  u = 1.0f - v - w;
}


bool isfinite(vgl_point_3d<double> const&p) {
  return vnl_math::isfinite(p.x()) && vnl_math::isfinite(p.y()) && vnl_math::isfinite(p.z());
}

bool isfinite(vgl_point_2d<double> const&p) {
  return vnl_math::isfinite(p.x()) && vnl_math::isfinite(p.y());
}

bool good_triangle(vgl_point_3d<double> const& a,
                   vgl_point_3d<double> const& b,
                   vgl_point_3d<double> const& c)
{
  double ab = (a - b).length();
  double bc = (b - c).length();
  double ca = (c - a).length();

  double min_side = ab;
  double max_side = ab;
  if (bc < min_side) {
    min_side = bc;
  }
  if (ca  < min_side) {
    min_side = ca;
  }
  if (bc > max_side) {
    max_side = bc;
  }
  if (ca > max_side) {
    max_side = ca;
  }
  const double max_triangle_side_ratio = 10.0;
  return max_side / min_side < max_triangle_side_ratio;
}


//: helper function to render a triangle into the heightmap
void render_triangle(vgl_point_2d<double> const& a2d,
                     vgl_point_2d<double> const& b2d,
                     vgl_point_2d<double> const& c2d,
                     vgl_point_3d<double> const& a3d,
                     vgl_point_3d<double> const& b3d,
                     vgl_point_3d<double> const& c3d,
                     vil_image_view<float> &heightmap)
{
  // make sure all points are valid
  if (!(isfinite(a2d) && isfinite(b2d) && isfinite(c2d) &&
        isfinite(a3d) && isfinite(b3d) && isfinite(c3d)) ){
    return;
  }
  // make sure triangle is "good"
  // Bad triangles are very long and skinny, usually as a result of one bad/noisy point
  if (!good_triangle(a3d, b3d, c3d)) {
    return;
  }

  vgl_triangle_scan_iterator<double> tri_scanner;
  tri_scanner.a.x = a2d.x();
  tri_scanner.a.y = a2d.y();
  tri_scanner.b.x = b2d.x();
  tri_scanner.b.y = b2d.y();
  tri_scanner.c.x = c2d.x();
  tri_scanner.c.y = c2d.y();
  tri_scanner.reset();
  while(tri_scanner.next()) {
    int v = tri_scanner.scany();
    if (v < 0) {
      continue;
    }
    if (v >= heightmap.nj()) {
      break;
    }
    int u0 = tri_scanner.startx();
    if (u0 < 0) {
      u0 = 0;
    }
    int u1 = tri_scanner.endx();
    if (u1 >= static_cast<int>(heightmap.ni())) {
      u1 = heightmap.ni() - 1;
    }
    for (int u=u0; u<=u1; ++u) {
      // interpolate 3D point and fill in image
      double wa, wb, wc;
      compute_barycentric(a2d, b2d, c2d,
                          vgl_point_2d<double>(u,v),
                          wa, wb, wc);
      double z = wa*a3d.z() + wb*b3d.z() + wc*c3d.z();

      // only replace existing valid z value if the new value is higher
      if ( (!vnl_math::isfinite(heightmap(u,v))) || (z > heightmap(u,v)) ) {
        heightmap(u,v) = z;
      }
    }
  }
}



template<class CAM_T>
vil_image_view<float>
bpgl_heightmap_from_disparity(CAM_T const& cam1, CAM_T const& cam2,
                              vil_image_view<float> disparity, vgl_box_3d<double> heightmap_bounds,
                              double ground_sample_distance)
{
  // convert disparity to set of 3D points
  vil_image_view<float> triangulated = bpgl_3d_from_disparity(cam1, cam2, disparity);
  const int triangulated_ni = triangulated.ni();
  const int triangulated_nj = triangulated.nj();

  const float min_z = heightmap_bounds.min_z();
  const float max_z = heightmap_bounds.max_z();

  const vgl_point_2d<double> bad_point_2d(NAN, NAN);
  const vgl_point_3d<double> bad_point_3d(NAN, NAN, NAN);

  // set ni,nj such that image contains all samples within bounds, inclusive
  size_t ni = static_cast<unsigned>(std::floor(heightmap_bounds.width() / ground_sample_distance + 1));
  size_t nj = static_cast<unsigned>(std::floor(heightmap_bounds.height() / ground_sample_distance + 1));
  vil_image_view<float> hmap(ni,nj);
  hmap.fill(NAN);

  // project each 3D point into the heightmap image
  vbl_array_2d<vgl_point_3d<double> > pts3d(triangulated_ni, triangulated_nj);
  vbl_array_2d<vgl_point_2d<double> > pts2d(triangulated_ni, triangulated_nj);
  for (int j=0; j<triangulated_nj; ++j) {
    for (int i=0; i<triangulated_ni; ++i) {
      vgl_point_3d<double> p3d(triangulated(i,j,0),
                                triangulated(i,j,1),
                                triangulated(i,j,2));
      if ( (p3d.z() >= min_z) && (p3d.z() <= max_z) ) {
        vgl_vector_3d<double> p3d_hmap = (p3d - heightmap_bounds.min_point()) / ground_sample_distance;
        pts2d(i,j) = vgl_point_2d<double>(p3d_hmap.x(), nj - p3d_hmap.y() - 1);
        pts3d(i,j) = p3d;
      }
      else {
        pts2d(i,j) = bad_point_2d;
        pts3d(i,j) = bad_point_3d;
      }
    }
  }


  for (int j=1; j<triangulated_nj; ++j) {
    for (int i=1; i<triangulated_ni; ++i) {
      // triangle 1
      render_triangle(pts2d(i-1,j-1),
                      pts2d(i, j-1),
                      pts2d(i,j),
                      pts3d(i-1, j-1),
                      pts3d(i, j-1),
                      pts3d(i,j),
                      hmap);
      // triangle 2
      render_triangle(pts2d(i-1,j),
                      pts2d(i-1, j-1),
                      pts2d(i,j),
                      pts3d(i-1, j),
                      pts3d(i-1, j-1),
                      pts3d(i,j),
                      hmap);

      // Note that the first two triangles are enough to cover the full image,
      // but "double-covering" with the next two triangles reduces the damage that
      // a single noisy point causes to its neighborhood.

      // triangle 3
      render_triangle(pts2d(i-1,j-1),
                      pts2d(i, j-1),
                      pts2d(i-1,j),
                      pts3d(i-1, j-1),
                      pts3d(i, j-1),
                      pts3d(i-1,j),
                      hmap);
      // triangle 4
      render_triangle(pts2d(i-1,j),
                      pts2d(i, j-1),
                      pts2d(i,j),
                      pts3d(i-1, j),
                      pts3d(i, j-1),
                      pts3d(i,j),
                      hmap);

    }
  }
  // final bounds check to remove outliers
  for (int j=0; j<nj; ++j) {
    for (int i=0; i<ni; ++i) {
      if ((hmap(i,j) < min_z) || (hmap(i,j) > max_z)) {
        hmap(i,j) = NAN;
      }
    }
  }
  return hmap;
}


#define BPGL_HEIGHTMAP_FROM_DISPARITY_INSTANIATE(CAM_T) \
template vil_image_view<float> \
bpgl_heightmap_from_disparity<CAM_T>(CAM_T const& cam1, CAM_T const& cam2, \
                                     vil_image_view<float> disparity, \
                                     vgl_box_3d<double> heightmap_bounds, \
                                     double ground_sample_distance)

#endif
