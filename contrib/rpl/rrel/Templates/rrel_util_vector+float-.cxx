#include <iostream>
#include <vector>
#include <vcl_compiler.h>
#include <rrel/rrel_util.hxx>

// Apply explicit instantiation
typedef std::vector<float>::iterator Iter;
RREL_UTIL_INSTANTIATE_RAN_ITER(float, Iter);
